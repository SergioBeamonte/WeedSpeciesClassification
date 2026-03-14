import os
# Apagar los warnings molestos de TensorFlow y oneDNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Oculta los mensajes INFO y WARNING de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Apaga las operaciones custom de oneDNN

import cv2
import numpy as np
import nmslib
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image

#from keras.applications.resnet50 import ResNet50, preprocess_input
#from keras.preprocessing import image

class ToyCBIRSystem:
    def __init__(self, index_file="image_index_v2.nmslib", metadata_file="metadata_v2.pkl"):
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.image_paths = []
        
        # 1. CNN Model (ResNet50) - 2048 dim embeddings
        self.cnn_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        
        # Dimensions : 2048 (CNN) + 96 (Color) + 324 (HOG) + 4 (Canny/Sobel) = 2472
        # self.dimension = 2472 

        # Dimensions : 2048 (CNN) + 96 (Color) + 324 (HOG) + 26 (LBP) + 32 (ORB) = 2526
        self.dimension = 2526
        
        # Init NMSLIB index HNSW and L2 distance
        self.index = nmslib.init(method='hnsw', space='l2') #, data_type=nmslib.DataType.FLOAT)

    def extract_raw_features(self, img_path):
        """Extrae los descriptores por separado y los devuelve en un diccionario."""
        img = cv2.imread(img_path)
        if img is None: return None

        if img.shape[1] > 100:
            img = img[:, 30:-70]
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img_res = cv2.resize(img_rgb, (514, 514))
        img_res = cv2.resize(img_rgb, (224, 224))


        img_gray = cv2.cvtColor(img_res, cv2.COLOR_RGB2GRAY)

        # --- A. CNN FEATURES ---
        x = image.img_to_array(img_res)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat_cnn = self.cnn_model.predict(x, verbose=0).flatten()
        feat_cnn /= (np.linalg.norm(feat_cnn) + 1e-7)

        # --- B. COLOR HISTOGRAM ---
        img_hsv = cv2.cvtColor(img_res, cv2.COLOR_RGB2HSV)
        hist_h = cv2.calcHist([img_hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([img_hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([img_hsv], [2], None, [32], [0, 256])
        feat_color = np.concatenate([hist_h, hist_s, hist_v]).flatten()
        feat_color /= (np.linalg.norm(feat_color) + 1e-7)

        # --- C. HOG ---
        feat_hog = hog(img_gray, orientations=9, pixels_per_cell=(32, 32), 
                       cells_per_block=(2, 2), feature_vector=True)
        feat_hog /= (np.linalg.norm(feat_hog) + 1e-7)

        # --- D. CANNY + SOBEL ---
        edges = cv2.Canny(img_gray, 100, 200)
        sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
        sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
        angles = np.abs(np.arctan2(sobely, sobelx) * 180 / np.pi)
        feat_canny = np.histogram(angles[edges > 0], bins=4, range=(0, 180))[0].astype('float32')
        feat_canny /= (np.linalg.norm(feat_canny) + 1e-7)

        # --- E. LBP (Textura Local) ---
        # Ampliamos el radio a 3 para capturar texturas más grandes (nervaduras, rugosidad real)
        radius = 3  
        n_points = 24 # Siempre suele ser 8 * radio
        
        # method='uniform' es clave: reduce el ruido y hace que el vector final sea de solo 26 dimensiones
        lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
        
        n_bins = int(lbp.max() + 1)
        feat_lbp, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        
        feat_lbp = feat_lbp.astype('float32')
        # NORMALIZACIÓN OBLIGATORIA
        feat_lbp /= (np.linalg.norm(feat_lbp) + 1e-7)

        # --- F. ORB (Puntos Característicos) ---
        orb = cv2.ORB_create(nfeatures=500)
        # Encontramos los keypoints y descriptores
        keypoints, descriptors = orb.detectAndCompute(img_gray, None)
        
        # Truco sucio pero efectivo: Average Pooling para tener un vector de tamaño fijo
        if descriptors is not None:
            feat_orb = np.mean(descriptors, axis=0).astype('float32')
        else:
            # Si la imagen es tan mala que no pilla ni un punto, devolvemos ceros
            feat_orb = np.zeros(32, dtype='float32')
            
        # Normalización (crucial)
        feat_orb /= (np.linalg.norm(feat_orb) + 1e-7)



        return {
            'cnn': feat_cnn.astype('float32'),
            'color': feat_color.astype('float32'),
            'hog': feat_hog.astype('float32'),
            # 'canny': feat_canny.astype('float32'),
            'lbp': feat_lbp.astype('float32'),
            'orb': feat_orb.astype('float32') 
        }
    
    def extract_features(self, img_path):
        """Usa raw_features, aplica pesos y los concatena para la búsqueda en NMSLIB."""
        raw_feats = self.extract_raw_features(img_path)
        if raw_feats is None: return None
        
        # --- SISTEMA DE PESOS ---
        w_cnn = 1.0
        w_color = 0.1  
        w_hog = 1.0    
        # w_canny = 1.0
        w_lbp = 1.0
        w_orb = 0.8 
        
        return np.concatenate([
            raw_feats['cnn'] * w_cnn, 
            raw_feats['color'] * w_color, 
            raw_feats['hog'] * w_hog, 
            # raw_feats['canny'] * w_canny
            raw_feats['lbp'] * w_lbp                            
        ])


    def visualize_raw_stats(self, query_path, results):
        """Muestra la imagen junto con los gráficos de sus 5 descriptores."""
        n = len(results)
        # Matriz: (n+1) filas x 6 columnas
        fig, axes = plt.subplots(n + 1, 6, figsize=(30, 5 * (n + 1)))
        
        def plot_row(row_idx, path, title_text):
            feats = self.extract_raw_features(path)
            if feats is None: return
            
            # Col 0: Imagen
            axes[row_idx, 0].imshow(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
            axes[row_idx, 0].set_title(title_text)
            axes[row_idx, 0].axis('off')
            
            # Col 1: CNN (Vector de 2048 dims)
            axes[row_idx, 1].plot(feats['cnn'], color='blue', alpha=0.7)
            axes[row_idx, 1].set_title("CNN (2048 dims)")
            axes[row_idx, 1].set_ylim(0, max(feats['cnn']) * 1.1 if max(feats['cnn']) > 0 else 1)
            
            # Col 2: Color HSV
            axes[row_idx, 2].plot(feats['color'], color='teal')
            axes[row_idx, 2].fill_between(range(len(feats['color'])), feats['color'], color='teal', alpha=0.3)
            axes[row_idx, 2].set_title("Color HSV (96 bins)")
            
            # Col 3: HOG (Estructura global)
            axes[row_idx, 3].plot(feats['hog'], color='green', alpha=0.7)
            axes[row_idx, 3].set_title(f"HOG ({len(feats['hog'])} dims)")
            
            # # Col 4: Canny (4 direcciones)
            # bars = axes[row_idx, 4].bar(['0°', '45°', '90°', '135°'], feats['canny'], color='orange')
            # axes[row_idx, 4].set_title("Canny (4 bins)")
            # axes[row_idx, 4].set_ylim(0, 1.0)
            # # Valores sobre las barras de Canny
            # for bar in bars:
            #     yval = bar.get_height()
            #     axes[row_idx, 4].text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center')

            # Col 4: LBP (Textura Local - 26 bins)
            axes[row_idx, 4].plot(feats['lbp'], color='orange')
            axes[row_idx, 4].fill_between(range(len(feats['lbp'])), feats['lbp'], color='orange', alpha=0.3)
            axes[row_idx, 4].set_title(f"LBP Textura ({len(feats['lbp'])} bins)")
            
            # Ajustamos el límite Y dinámicamente para que no quede aplastado
            max_val = max(feats['lbp'])
            axes[row_idx, 4].set_ylim(0, max_val * 1.1 if max_val > 0 else 1.0)

            # Col 5: ORB (Puntos característicos - 32 bins)
            axes[row_idx, 5].plot(feats['orb'], color='purple')
            axes[row_idx, 5].fill_between(range(len(feats['orb'])), feats['orb'], color='purple', alpha=0.3)
            axes[row_idx, 5].set_title(f"ORB Keypoints ({len(feats['orb'])} dims)")
            
            max_val_orb = max(feats['orb'])
            axes[row_idx, 5].set_ylim(0, max_val_orb * 1.1 if max_val_orb > 0 else 1.0)

        # --- Fila 0: CONSULTA ---
        query_plant = os.path.basename(query_path)[:5]
        plot_row(0, query_path, f"CONSULTA\nPlanta: {query_plant}")

        # --- Filas 1 a N: RESULTADOS ---
        for i, (path, total_dist) in enumerate(results):
            res_plant = os.path.basename(path)[:5]
            plot_row(i + 1, path, f"Rank {i+1} | Dist: {total_dist:.3f}\nPlanta: {res_plant}")

        plt.tight_layout()
        plt.show()

    def index_folder(self, folder_path):
        print(f"Indexing folder: {folder_path}...")
        features_list = []
        self.image_paths = []
        
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]

        for filename in tqdm(files):
            path = os.path.join(folder_path, filename)
            feat = self.extract_features(path)
            if feat is not None:
                features_list.append(feat)
                self.image_paths.append(path)

        if not features_list: return

        # Loading data in NMSLIB
        data_matrix = np.array(features_list)
        self.index.addDataPointBatch(data_matrix)
        
        # Creating index HNSW
        print("Building HNSW graph...")
        self.index.createIndex({'M': 16, 'post': 2, 'efConstruction': 200}, print_progress=True)

        # Saving index to disk
        self.index.saveIndex(self.index_file, save_data=True)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.image_paths, f)
        print(f"Completado. {len(self.image_paths)} imágenes listas.")

    def load_index(self):
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            self.index.loadIndex(self.index_file, load_data=True)
            with open(self.metadata_file, 'rb') as f:
                self.image_paths = pickle.load(f)
            return True
        return False

    def search(self, query_path, top_k=5):
        # create the descriptors for input query image
        query_feat = self.extract_features(query_path)
        if query_feat is None: return []
        
        # k-NN search 1:N input query among all indexed images  
        indices, distances = self.index.knnQuery(query_feat, k=top_k+1) # +1 because the query image itself will be the closest (distance=0)
        # Remove the query image itself from the results
        indices = indices[1:]
        distances = distances[1:]
        return [(self.image_paths[idx], distances[i]) for i, idx in enumerate(indices)]

    def visualize(self, query_path, results):
        n = len(results)
        fig, axes = plt.subplots(1, n + 1, figsize=(20, 5))
        
        # Extraer el tipo de planta de la consulta
        query_filename = os.path.basename(query_path)
        query_plant = query_filename[:5]    
        
        # Plot de la imagen de Consulta
        axes[0].imshow(cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"IMAGEN CONSULTA\nPlanta: {query_plant}")
        axes[0].axis('off')

        # Plot de los Resultados
        for i, (path, dist) in enumerate(results):
            # Extraer el tipo de planta del resultado
            res_filename = os.path.basename(path)
            res_plant = res_filename[:5]
            
            axes[i+1].imshow(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
            axes[i+1].set_title(f"Rank {i+1}\nDist: {dist:.3f}\nPlanta: {res_plant}")
            axes[i+1].axis('off')
        
        plt.tight_layout()
        plt.show()

# --- USING THE TOY CBIR ---
if __name__ == "__main__":
    cbir = ToyCBIRSystem()

    # 1. Intentamos cargar el índice guardado previamente
    if not cbir.load_index():
        print("Índice no encontrado. Extrayendo características y creando uno nuevo...")
        # Solo indexa si load_index() devuelve False
        cbir.index_folder("./data/small_jpegs") 
        cbir.load_index() # Lo cargamos en memoria tras crearlo
    else:
        print("¡Índice cargado desde el disco instantáneamente!")

    # 2. Búsqueda
    query = "data\\small_jpegs\\PLAMA_134829_2021Y11M29D_13H03M22S_img.jpeg" # Asegúrate de que esta variable esté definida
    if os.path.exists(query):
        res = cbir.search(query, top_k=5)
        cbir.visualize(query, res)
        cbir.visualize_raw_stats(query, res)
    else:
        print(f"No se encontró la imagen de consulta: {query}")
from ftplib import FTP
from pathlib import Path
import pandas as pd
import argparse
from tqdm import tqdm
import io
import zipfile
import shutil

NUMBER_OF_FILES = 10

def get_list_to_download(df, lst, column):
    df2 = df[df[column].isin(lst)]
    df3 = df2.groupby(["label_id", "tray_id"]).first().reset_index()
    df4 = df3["filename"].str.split("/", expand=True)
    to_download = list(df4[0] + "/" + df4[1] + ".zip")
    return to_download


def download_gt_file(save_path):
    """
    Download gt.csv file if it does not exist
    :param save_path: path to save this file to
    :return:
    """
    ftp = FTP('dataserv.ub.tum.de')
    ftp.login('m1717366', 'm1717366')
    save_file_path = save_path / "gt.csv"
    if not list(save_path.glob("gt.csv")):
        print("Downloading gt.csv...")
        with open(save_file_path, 'wb') as f:
            ftp.retrbinary('RETR ' + "gt.csv", f.write)
    return


def download_all_files_with_segmentation_masks(save_path, image_type, species=None):
    print("Downloading all files with segmentation masks...")
    download_gt_file(save_path)
    df = pd.read_csv(save_path / "gt.csv")
    tray_ids = [132801, 103814, 136813, 104806, 109811, 108807, 107907, 131803, 114905, 116814, 124832, 118934, 120902,
                139837]
    filenames = get_list_to_download(df, tray_ids, "tray_id")
    folders = [image_type, "masks/panoptic_segmentation", "masks/semantic_segmentation"]
    for folder in folders:
        trange = tqdm(filenames, total=len(filenames))
        for fname in trange:
            download_file(save_path, folder, fname)
            trange.set_description_str(folder+"/"+fname)
    return


def download_species(save_path, image_type, species):
    download_gt_file(save_path)
    print(f"Downloading {species}...")
    df = pd.read_csv(save_path / "gt.csv")
    filenames = get_list_to_download(df, species, "label_id")

    # Solo me quedo con unas cuantas
    filenames = filenames[:NUMBER_OF_FILES]

    trange = tqdm(filenames, total=len(filenames))
    for fname in trange:
            download_file(save_path, image_type, fname)
            trange.set_description_str(image_type+"/"+fname)
    return

def download_trays(save_path, image_type, trays):
    download_gt_file(save_path)
    print(f"Downloading {trays}...")
    df = pd.read_csv(save_path / "gt.csv")
    trays = [int(tray) for tray in trays]  # cast to int
    filenames = get_list_to_download(df, trays, "tray_id")
    trange = tqdm(filenames, total=len(filenames))
    for fname in trange:
            download_file(save_path, image_type, fname)
            trange.set_description_str(image_type+"/"+fname)
    return

def download_varied(save_path, image_type, dummy=None):
    download_gt_file(save_path)
    df = pd.read_csv(save_path / "gt.csv")
    
    # Agrupamos para no coger todas de la misma planta y asegurar variedad
    df_unique = df.groupby(["label_id", "tray_id"]).first().reset_index()
    
    # Pillamos aleatorias 
    df_sample = df_unique.sample(n=NUMBER_OF_FILES, random_state=42)
    
    df4 = df_sample["filename"].str.split("/", expand=True)
    filenames = list(df4[0] + "/" + df4[1] + ".zip")
    
    trange = tqdm(filenames, total=NUMBER_OF_FILES)
    for fname in trange:
        download_file(save_path, image_type, fname)
        trange.set_description_str(image_type+"/"+fname)
    return


def download_file(save_path, folder, fname):
    ftp = FTP('dataserv.ub.tum.de')
    ftp.login('m1717366', 'm1717366')
    
    # fname viene como "carpeta1/carpeta2.zip"
    save_folder = Path(save_path) / folder
    save_folder.mkdir(exist_ok=True, parents=True)
    
    # Creamos un espacio en la memoria RAM para el zip
    zip_buffer = io.BytesIO()
    
    try:
        # Descargamos el zip directamente a la RAM
        ftp.retrbinary('RETR ' + folder + "/" + fname, zip_buffer.write)
    except Exception as e:
        print(f"\nError descargando {fname}: {e}")
        return

    try:
        # Leemos el zip desde la RAM
        with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
            archivos = zip_ref.infolist()
            
            # Filtramos para ignorar carpetas, solo queremos archivos
            archivos = [f for f in archivos if not f.is_dir()]
            
            if not archivos:
                return

            # Ordenamos alfabéticamente por el NOMBRE del archivo
            archivos.sort(key=lambda x: x.filename)
            
            # Cogemos la penúltima (índice -2). Si por lo que sea solo hay 1 archivo, pillamos ese.
            if len(archivos) >= 2:
                target = archivos[-2]
            else:
                target = archivos[0] 
                
            # Extraemos SOLO ese archivo físicamente al disco duro
            ruta_extraida = Path(zip_ref.extract(target, path=save_folder))
            
            # Forzamos que la ruta final sea directamente dentro de "jpegs" (o la carpeta principal)
            ruta_final = save_folder / ruta_extraida.name
            
            # Si se ha extraído en una subcarpeta, la sacamos de ahí
            if ruta_extraida != ruta_final:
                # Nos cargamos el archivo destino si ya había algo para que shutil.move no tire error
                if ruta_final.exists():
                    ruta_final.unlink()
                
                shutil.move(str(ruta_extraida), str(ruta_final))
                
                # Intentamos borrar la carpeta vacía que deja atrás, si peta porque hay más mierda, pasamos olímpicamente
                try:
                    ruta_extraida.parent.rmdir()
                except OSError:
                    pass
            
    except zipfile.BadZipFile:
        print(f"\nEl archivo {fname} está corrupto o vacío.")
        
    # Al terminar la función, 'zip_buffer' se elimina de la RAM automáticamente.
    return

FUNCTION_MAP = {'species': download_species,
                'masks': download_all_files_with_segmentation_masks,
                'trays': download_trays,
                'varied': download_varied}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('download', choices=FUNCTION_MAP.keys())
    parser.add_argument('-save_path', type=str, default='/local2')
    parser.add_argument('-files', type=str, default='ARTVU,CHEAL', help="Comma separated list of EPPO codes or trays IDS to download")
    parser.add_argument('-img_type', type=str, default='jpegs', help="Type of the images: 'jpegs' or 'pngs'")
    args = parser.parse_args()
    files = args.files.split(",")
    files = [spec.strip() for spec in files]
    func = FUNCTION_MAP[args.download]
    func(Path(args.save_path), args.img_type, files)

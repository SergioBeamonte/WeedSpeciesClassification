from ftplib import FTP
from pathlib import Path
import pandas as pd
import argparse
import io
import zipfile
import shutil

NUMBER_OF_FILES = 20

def get_list_to_download(df, lst, column):
    df2 = df[df[column].isin(lst)]
    df3 = df2.groupby(["label_id", "tray_id"]).first().reset_index()
    df4 = df3["filename"].str.split("/", expand=True)
    to_download = list(df4[0] + "/" + df4[1] + ".zip")
    return to_download

def download_gt_file(save_path):
    ftp = FTP('dataserv.ub.tum.de')
    ftp.login('m1717366', 'm1717366')
    save_file_path = save_path / "gt.csv"
    if not list(save_path.glob("gt.csv")):
        print("Downloading gt.csv...", flush=True)
        with open(save_file_path, 'wb') as f:
            ftp.retrbinary('RETR ' + "gt.csv", f.write)
    return

def download_all_files_with_segmentation_masks(save_path, image_type, species=None):
    print("Downloading all files with segmentation masks...", flush=True)
    download_gt_file(save_path)
    df = pd.read_csv(save_path / "gt.csv")
    tray_ids = [132801, 103814, 136813, 104806, 109811, 108807, 107907, 131803, 114905, 116814, 124832, 118934, 120902,
                139837]
    filenames = get_list_to_download(df, tray_ids, "tray_id")
    folders = [image_type, "masks/panoptic_segmentation", "masks/semantic_segmentation"]
    for folder in folders:
        for fname in filenames:
            download_file(save_path, folder, fname)
    return

def download_species(save_path, image_type, species):
    download_gt_file(save_path)
    print(f"Downloading {species}...", flush=True)
    df = pd.read_csv(save_path / "gt.csv")
    filenames = get_list_to_download(df, species, "label_id")

    filenames = filenames[:NUMBER_OF_FILES]

    for fname in filenames:
        download_file(save_path, image_type, fname)
    return

def download_trays(save_path, image_type, trays):
    download_gt_file(save_path)
    print(f"Downloading {trays}...", flush=True)
    df = pd.read_csv(save_path / "gt.csv")
    trays = [int(tray) for tray in trays] 
    filenames = get_list_to_download(df, trays, "tray_id")
    for fname in filenames:
        download_file(save_path, image_type, fname)
    return

def download_varied(save_path, image_type, dummy=None):
    download_gt_file(save_path)
    df = pd.read_csv(save_path / "gt.csv")
    
    df_unique = df.groupby(["label_id", "tray_id"]).first().reset_index()
    df_sample = df_unique.sample(n=NUMBER_OF_FILES, random_state=42)
    
    df4 = df_sample["filename"].str.split("/", expand=True)
    filenames = list(df4[0] + "/" + df4[1] + ".zip")
    
    for fname in filenames:
        download_file(save_path, image_type, fname)
    return


def download_file(save_path, folder, fname):
    small_jpegs_dir = Path(save_path) / "small_jpegs"
    prefix = Path(fname).name[:6]
    
    if small_jpegs_dir.exists():
        # Comprobamos si los caracteres de la posición 6 a la 12 del archivo existente coinciden con el prefijo
        if any(f.name[6:12] == prefix for f in small_jpegs_dir.iterdir() if f.is_file()):
            print(f"[SALTADO] Ya existe archivo coincidente con {prefix} en las posiciones 6-12 en small_jpegs", flush=True)
            return

    print(f"\n[DESCARGANDO] Zip: {fname} en la carpeta {folder}...", flush=True)

    ftp = FTP('dataserv.ub.tum.de')
    ftp.login('m1717366', 'm1717366')
    
    save_folder = Path(save_path) / folder
    save_folder.mkdir(exist_ok=True, parents=True)
    
    zip_buffer = io.BytesIO()
    
    try:
        ftp.retrbinary('RETR ' + folder + "/" + fname, zip_buffer.write)
    except Exception as e:
        print(f"Error descargando {fname}: {e}", flush=True)
        return

    try:
        with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
            archivos_brutos = zip_ref.infolist()
            archivos = [f for f in archivos_brutos if not f.is_dir()]
            
            if not archivos:
                return

            archivos.sort(key=lambda x: x.filename)
            
            if len(archivos) >= 2:
                target = archivos[-2]
            else:
                target = archivos[0] 
                
            ruta_extraida = Path(zip_ref.extract(target, path=save_folder))
            ruta_final = save_folder / ruta_extraida.name
            
            if ruta_extraida != ruta_final:
                if ruta_final.exists():
                    ruta_final.unlink()
                
                shutil.move(str(ruta_extraida), str(ruta_final))
                
                try:
                    ruta_extraida.parent.rmdir()
                except OSError:
                    pass
            
    except zipfile.BadZipFile:
        print(f"El archivo {fname} está corrupto o vacío.", flush=True)
        
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
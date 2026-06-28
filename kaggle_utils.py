import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset(dataset, path):
    if os.path.exists(path) and len(os.listdir(path)) > 0:
        print(f"Data already exists at {path}. Skipping download.")
        return True

    print(f"Downloading dataset {dataset} to {path}...")
    api = KaggleApi()
    api.authenticate()
    os.makedirs(path, exist_ok=True)
    api.dataset_download_files(dataset, path=path, unzip=True)
    print(f"Download of {dataset} complete.")
    return True

def download_manifold_data():
    return download_dataset('hichambedrani/stratos-manifold-v4', './kaggle_data/stratos_manifold')

def download_all_resources():
    resources = [
        ('hichambedrani/stratos-manifold-v4', './kaggle_data/stratos_manifold'),
        ('hichambedrani/stratoscot', './kaggle_data/stratoscot'),
        ('hichambedrani/stratos-omega-manifold-v3', './kaggle_data/omega_manifold'),
        ('hichambedrani/fso-manifold', './kaggle_data/fso_manifold'),
        ('hichambedrani/precision-system-v3-data', './kaggle_data/precision_data')
    ]
    for ds, path in resources:
        download_dataset(ds, path)

if __name__ == "__main__":
    download_all_resources()

import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_manifold_data(dataset='hichambedrani/stratos-manifold-v4', path='./kaggle_data'):
    """
    Downloads and unzips the specified Kaggle dataset.
    Relies on KAGGLE_API_TOKEN being set in the environment.
    """
    if 'KAGGLE_API_TOKEN' not in os.environ:
        print("Warning: KAGGLE_API_TOKEN not found in environment.")
        # We don't raise error here to allow fallback if data already exists
        return False

    if os.path.exists(path) and len(os.listdir(path)) > 0:
        print(f"Data already exists at {path}. Skipping download.")
        return True

    print(f"Downloading dataset {dataset} to {path}...")
    api = KaggleApi()
    api.authenticate()

    os.makedirs(path, exist_ok=True)
    api.dataset_download_files(dataset, path=path, unzip=True)
    print("Download and unzip complete.")
    return True

if __name__ == "__main__":
    download_manifold_data()

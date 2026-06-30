



import os
import zipfile
import time
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset(dataset, path, retries=3):
    if os.path.exists(path) and len(os.listdir(path)) > 0:
        print(f"Data already exists at {path}. Skipping download.")
        return True

    print(f"Downloading dataset {dataset} to {path}...")
    api = KaggleApi()
    api.authenticate()
    os.makedirs(path, exist_ok=True)

    for i in range(retries):
        try:
            api.dataset_download_files(dataset, path=path, unzip=True)
            print(f"Download of {dataset} complete.")
            return True
        except Exception as e:
            print(f"Attempt {i+1} failed to download {dataset}: {e}")
            if i < retries - 1:
                time.sleep(5)
            else:
                raise e
    return False

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

class KaggleSearch:
    """
    Search utility for Kaggle Datasets and Models.
    """
    def __init__(self):
        self.api = KaggleApi()
        self.api.authenticate()

    def search_datasets(self, query):
        print(f"Searching for datasets matching: {query}")
        datasets = self.api.dataset_list(search=query)
        for ds in datasets:
            print(f"Dataset: {ds.ref} | Title: {ds.title}")
        return datasets

    def search_models(self, query):
        print(f"Searching for models matching: {query}")
        try:
            models = self.api.model_list(search=query)
            for model in models:
                print(f"Model: {model.ownerSlug}/{model.slug} | Title: {model.title}")
            return models
        except AttributeError:
            print("Model search not supported in this Kaggle API version.")
            return []

    def discover_and_download_resources(self, query, base_path='./kaggle_data/discovered'):
        print(f"Discovering and downloading resources for: {query}")
        datasets = self.search_datasets(query)
        downloaded_paths = []
        for ds in datasets[:3]: # Limit to top 3
            path = os.path.join(base_path, ds.ref.replace('/', '_'))
            if download_dataset(ds.ref, path):
                downloaded_paths.append(path)
        return downloaded_paths

if __name__ == "__main__":
    search = KaggleSearch()
    search.search_datasets("manifold")

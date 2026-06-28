import os
import numpy as np
import torch
from kaggle_utils import download_manifold_data

class ManifoldLoader:
    """
    Utility to load weights from the Stratos Manifold dataset.
    Weights are stored as (1024,) .npy files.
    """
    def __init__(self, directory='kaggle_data/stratos_manifold'):
        # Ensure data is available
        download_manifold_data()

        self.directory = directory
        if not os.path.exists(directory):
            # Try the parent directory if the unzip structure is different
            parent = os.path.dirname(directory)
            if os.path.exists(parent) and len(os.listdir(parent)) > 0:
                # If only one subdirectory exists, use it
                subs = [d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d))]
                if len(subs) == 1:
                    self.directory = os.path.join(parent, subs[0])
                else:
                    self.directory = parent
            else:
                raise FileNotFoundError(f"Could not find manifold data directory at {directory}")

        self.weight_files = sorted([f for f in os.listdir(self.directory) if f.startswith('weight_math')])
        self.lib_files = sorted([f for f in os.listdir(self.directory) if f.startswith('lib_logic_math')])

    def load_weights(self, num_weights, edge_dim=32, node_dim=32):
        if num_weights > len(self.weight_files):
            num_weights = len(self.weight_files)

        weights = []
        for i in range(num_weights):
            data = np.load(os.path.join(self.directory, self.weight_files[i]))
            weights.append(data.reshape(edge_dim, node_dim))

        return torch.tensor(np.array(weights), dtype=torch.float32)

    def load_library(self, num_libs, edge_dim=32, node_dim=32):
        if num_libs > len(self.lib_files):
            num_libs = len(self.lib_files)

        libs = []
        for i in range(num_libs):
            data = np.load(os.path.join(self.directory, self.lib_files[i]))
            libs.append(data.reshape(edge_dim, node_dim))

        return torch.tensor(np.array(libs), dtype=torch.float32)

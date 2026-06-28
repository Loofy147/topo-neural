import os
import numpy as np
import torch
from kaggle_utils import download_manifold_data

class ManifoldLoader:
    def __init__(self, directory='kaggle_data/stratos_manifold'):
        download_manifold_data()

        self.directory = directory
        # Find the actual directory containing .npy files
        found_dir = None
        for root, dirs, files in os.walk('kaggle_data'):
            if any(f.endswith('.npy') for f in files):
                found_dir = root
                break

        if found_dir:
            self.directory = found_dir
            print(f"Manifold data found at: {self.directory}")
        else:
            raise FileNotFoundError(f"Could not find manifold data directory containing .npy files in kaggle_data")

        self.weight_files = sorted([f for f in os.listdir(self.directory) if f.startswith('weight_math')])
        self.lib_files = sorted([f for f in os.listdir(self.directory) if f.startswith('lib_logic_math')])
        self.ptr = 0

    def load_weights(self, num_weights, edge_dim=32, node_dim=32):
        end = min(self.ptr + num_weights, len(self.weight_files))
        actual_num = end - self.ptr

        if actual_num < num_weights:
            print(f"Warning: Only {actual_num} weights available from index {self.ptr}")

        weights = []
        for i in range(self.ptr, end):
            data = np.load(os.path.join(self.directory, self.weight_files[i]))
            weights.append(data.reshape(edge_dim, node_dim))

        self.ptr = end

        # Padding if not enough weights
        if len(weights) < num_weights:
            padding = [np.random.randn(edge_dim, node_dim) for _ in range(num_weights - len(weights))]
            weights.extend(padding)

        return torch.tensor(np.array(weights), dtype=torch.float32)

    def load_library(self, num_libs, edge_dim=32, node_dim=32):
        if num_libs > len(self.lib_files):
            num_libs = len(self.lib_files)

        libs = []
        for i in range(num_libs):
            data = np.load(os.path.join(self.directory, self.lib_files[i]))
            libs.append(data.reshape(edge_dim, node_dim))

        return torch.tensor(np.array(libs), dtype=torch.float32)

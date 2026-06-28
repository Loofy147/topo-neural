import os
import numpy as np
import torch
from kaggle_utils import download_all_resources

class ManifoldLoader:
    """
    Enhanced utility to load weights from multiple Stratos/Omega/FSO Manifold datasets.
    """
    def __init__(self, source='stratos', directory=None):
        download_all_resources()

        sources = {
            'stratos': 'kaggle_data/stratos_manifold',
            'omega': 'kaggle_data/omega_manifold',
            'fso': 'kaggle_data/fso_manifold'
        }

        base_dir = directory or sources.get(source)
        if not base_dir:
            raise ValueError(f"Unknown source: {source}")

        # Find the actual directory containing .npy files
        self.directory = self._find_npy_dir(base_dir)
        print(f"[{source.upper()} LOADER] Using directory: {self.directory}")

        self.weight_files = sorted([f for f in os.listdir(self.directory) if f.startswith('weight_')])
        self.lib_files = sorted([f for f in os.listdir(self.directory) if f.startswith('lib_')])
        self.ptr = 0

    def _find_npy_dir(self, start_path):
        for root, dirs, files in os.walk(start_path):
            if any(f.endswith('.npy') for f in files):
                return root
        raise FileNotFoundError(f"No .npy files found in {start_path}")

    def load_weights(self, num_weights, edge_dim=32, node_dim=32):
        end = min(self.ptr + num_weights, len(self.weight_files))
        actual_num = end - self.ptr

        weights = []
        for i in range(self.ptr, end):
            data = np.load(os.path.join(self.directory, self.weight_files[i]))
            # Flatten then reshape to fit requested dimensions
            flat_data = data.flatten()
            target_size = edge_dim * node_dim
            if flat_data.size >= target_size:
                reshaped = flat_data[:target_size].reshape(edge_dim, node_dim)
            else:
                # Pad with zeros if necessary
                padded = np.zeros(target_size)
                padded[:flat_data.size] = flat_data
                reshaped = padded.reshape(edge_dim, node_dim)
            weights.append(reshaped)

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
            flat_data = data.flatten()
            target_size = edge_dim * node_dim
            if flat_data.size >= target_size:
                reshaped = flat_data[:target_size].reshape(edge_dim, node_dim)
            else:
                padded = np.zeros(target_size)
                padded[:flat_data.size] = flat_data
                reshaped = padded.reshape(edge_dim, node_dim)
            libs.append(reshaped)

        return torch.tensor(np.array(libs), dtype=torch.float32)

import os
import numpy as np
import torch
from kaggle_utils import download_all_resources

class ManifoldLoader:
    """
    Enhanced utility to load weights from multiple Stratos/Omega/FSO Manifold datasets.
    Includes strict shape validation.
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
        if not os.path.exists(start_path):
             raise FileNotFoundError(f"Directory {start_path} does not exist.")
        for root, dirs, files in os.walk(start_path):
            if any(f.endswith('.npy') for f in files):
                return root
        raise FileNotFoundError(f"No .npy files found in {start_path}")

    def load_weights(self, num_weights, edge_dim=32, node_dim=32):
        end = min(self.ptr + num_weights, len(self.weight_files))
        actual_num = end - self.ptr

        weights = []
        target_size = edge_dim * node_dim

        for i in range(self.ptr, end):
            file_path = os.path.join(self.directory, self.weight_files[i])
            try:
                data = np.load(file_path)
            except Exception as e:
                raise IOError(f"Failed to load weight file {file_path}: {e}")

            # Flatten then reshape to fit requested dimensions
            flat_data = data.flatten()

            if flat_data.size < target_size:
                print(f"Warning: Data size {flat_data.size} in {self.weight_files[i]} is less than target {target_size}. Padding with zeros.")
                padded = np.zeros(target_size)
                padded[:flat_data.size] = flat_data
                reshaped = padded.reshape(edge_dim, node_dim)
            else:
                if flat_data.size > target_size:
                    print(f"Warning: Data size {flat_data.size} in {self.weight_files[i]} exceeds target {target_size}. Truncating.")
                reshaped = flat_data[:target_size].reshape(edge_dim, node_dim)
            weights.append(reshaped)

        self.ptr = end

        # If not enough weights, instead of random, we can raise error or pad.
        # Original code used random padding. Keeping it but with a message.
        if len(weights) < num_weights:
            print(f"Warning: Requested {num_weights} weights but only found {len(weights)}. Padding with random.")
            padding = [np.random.randn(edge_dim, node_dim) for _ in range(num_weights - len(weights))]
            weights.extend(padding)

        return torch.tensor(np.array(weights), dtype=torch.float32)

    def load_library(self, num_libs, edge_dim=32, node_dim=32):
        if num_libs > len(self.lib_files):
            print(f"Warning: Requested {num_libs} libraries but only found {len(self.lib_files)}.")
            num_libs = len(self.lib_files)

        libs = []
        target_size = edge_dim * node_dim
        for i in range(num_libs):
            file_path = os.path.join(self.directory, self.lib_files[i])
            try:
                data = np.load(file_path)
            except Exception as e:
                raise IOError(f"Failed to load library file {file_path}: {e}")

            flat_data = data.flatten()
            if flat_data.size < target_size:
                padded = np.zeros(target_size)
                padded[:flat_data.size] = flat_data
                reshaped = padded.reshape(edge_dim, node_dim)
            else:
                reshaped = flat_data[:target_size].reshape(edge_dim, node_dim)
            libs.append(reshaped)

        return torch.tensor(np.array(libs), dtype=torch.float32)

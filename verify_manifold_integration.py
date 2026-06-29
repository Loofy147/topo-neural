import os
import torch
from sheaf_nn import DeepSheafNetwork
from weights_loader import ManifoldLoader

def verify_full_version():
    print("--- Stratos Manifold Full Version Integration Test ---")

    # Grid dimensions
    H, W = 6, 6
    num_nodes = H * W
    edges = []
    for r in range(H):
        for c in range(W):
            u = r * W + c
            if c + 1 < W: edges.append((u, r * W + (c + 1)))
            if r + 1 < H: edges.append((u, (r + 1) * W + c))

    node_dim = 32
    edge_dim = 32

    # Initialize NCA model (more complex)
    print("Initializing DeepSheafNetwork (layer_type='nca')...")
    model = DeepSheafNetwork(num_nodes, edges, node_dim, edge_dim, num_layers=1, layer_type='nca')

    loader = ManifoldLoader()
    print("Loading manifold weights (including NCA specific logic)...")
    model.load_from_manifold(loader)

    H_initial = torch.randn(1, num_nodes, node_dim)

    print("Running forward pass...")
    with torch.no_grad():
        H_out = model(H_initial)

    print(f"Input shape: {H_initial.shape}")
    print(f"Output shape: {H_out.shape}")
    assert H_out.shape == H_initial.shape
    print("--- Full Version Verification Successful! ---")

if __name__ == "__main__":
    verify_full_version()

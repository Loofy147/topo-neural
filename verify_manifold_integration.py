import os
import torch
from sheaf_nn import DeepSheafNetwork
from weights_loader import ManifoldLoader

def verify():
    # Ensure KAGGLE_API_TOKEN is set for this demonstration if running from scratch
    if 'KAGGLE_API_TOKEN' not in os.environ:
        print("Note: KAGGLE_API_TOKEN not set in environment. This script will only work if data is already present.")

    # Grid dimensions
    H, W = 6, 6
    num_nodes = H * W

    # Construct grid edges
    edges = []
    for r in range(H):
        for c in range(W):
            u = r * W + c
            if c + 1 < W: # Horizontal
                v = r * W + (c + 1)
                edges.append((u, v))
            if r + 1 < H: # Vertical
                v = (r + 1) * W + c
                edges.append((u, v))

    # Manifold weights are (1024,), so we use dimensions that multiply to 1024
    node_dim = 32
    edge_dim = 32
    num_layers = 1

    print(f"--- Stratos Manifold Integration Test ---")
    print(f"Initializing DeepSheafNetwork with {num_layers} layer(s)...")
    model = DeepSheafNetwork(num_nodes, edges, node_dim, edge_dim, num_layers=num_layers)

    print("Initializing ManifoldLoader (this may trigger a download)...")
    try:
        loader = ManifoldLoader()
    except Exception as e:
        print(f"Failed to initialize loader: {e}")
        return

    print("Loading pre-trained manifold weights into the model...")
    model.load_from_manifold(loader)

    # Initial input features
    H_initial = torch.randn(1, num_nodes, node_dim)

    print("Running forward pass with manifold weights...")
    with torch.no_grad():
        H_out = model(H_initial)

    print(f"Input shape: {H_initial.shape}")
    print(f"Output shape: {H_out.shape}")

    assert H_out.shape == H_initial.shape
    print("--- Verification Successful! ---")

if __name__ == "__main__":
    verify()

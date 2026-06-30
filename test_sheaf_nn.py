import torch
from sheaf_nn import SheafDiffusionLayer, DeepSheafNetwork

def test_forward():
    # Define a simple triangle graph
    num_nodes = 3
    edges = [(0, 1), (1, 2), (2, 0)]
    node_dim = 4
    edge_dim = 2

    model = SheafDiffusionLayer(num_nodes, edges, node_dim, edge_dim)

    # Initialize node features
    H = torch.randn(2, num_nodes, node_dim) # Batch size 2

    # Forward pass
    H_out = model(H)

    print(f"Input shape: {H.shape}")
    print(f"Output shape: {H_out.shape}")

    assert H_out.shape == H.shape
    print("Forward pass successful!")

def test_stiefel():
    num_nodes = 2
    edges = [(0, 1)]
    node_dim = 4
    edge_dim = 4

    model = SheafDiffusionLayer(num_nodes, edges, node_dim, edge_dim)
    model.project_to_stiefel()

    W = model.W_maps
    # W^T * W should be identity
    WtW = torch.matmul(W.transpose(-1, -2), W)
    eye = torch.eye(node_dim).expand_as(WtW)

    diff = torch.norm(WtW - eye, p=float('inf'))
    print(f"Stiefel projection diff (inf-norm): {diff.item()}")
    # Using a slightly more relaxed threshold for float precision
    assert diff < 1e-2
    print("Stiefel projection successful!")

def test_deep_network():
    num_nodes = 3
    edges = [(0, 1), (1, 2)]
    node_dim = 4
    edge_dim = 2
    num_layers = 3

    model = DeepSheafNetwork(num_nodes, edges, node_dim, edge_dim, num_layers=num_layers)
    H = torch.randn(1, num_nodes, node_dim)
    H_out = model(H)

    assert H_out.shape == H.shape
    print("Deep network forward pass successful!")

if __name__ == "__main__":
    test_forward()
    test_stiefel()
    test_deep_network()

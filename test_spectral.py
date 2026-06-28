import torch
from sheaf_nn import SheafDiffusionLayer, SheafNCALayer
from spectral_topo import compute_sheaf_laplacian_spectral_gap

def test_spectral_gap():
    num_nodes = 4
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    node_dim = 2
    edge_dim = 2

    layer = SheafDiffusionLayer(num_nodes, edges, node_dim, edge_dim)
    # Ensure some connectivity
    layer.project_to_stiefel()

    gap = compute_sheaf_laplacian_spectral_gap(num_nodes, layer.edge_index, layer.W_maps, edge_dim, node_dim)
    print(f"Spectral Gap: {gap.item()}")
    assert gap >= 0

    # Gap should be backpropagatable
    gap.backward()
    assert layer.W_maps.grad is not None
    print("Spectral gap backprop successful!")

def test_nca_layer():
    num_nodes = 3
    edges = [(0, 1), (1, 2)]
    node_dim = 4
    edge_dim = 2

    layer = SheafNCALayer(num_nodes, edges, node_dim, edge_dim)
    H = torch.randn(1, num_nodes, node_dim)
    H_out = layer(H)

    assert H_out.shape == H.shape
    print("SheafNCA forward pass successful!")

if __name__ == "__main__":
    test_spectral_gap()
    test_nca_layer()

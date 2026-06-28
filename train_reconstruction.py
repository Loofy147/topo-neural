import torch
import torch.optim as optim
from sheaf_nn import DeepSheafNetwork
from topo_torch import relaxed_euler_torch
from spectral_topo import compute_sheaf_laplacian_spectral_gap

def train_reconstruction():
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

    node_dim = 8
    edge_dim = 4
    num_layers = 5

    # Initialize Sheaf NCA Model
    model = DeepSheafNetwork(num_nodes, edges, node_dim, edge_dim, num_layers=num_layers, layer_type='nca')
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Task: Reconstruction
    # Initial input has some "noise" or "holes"
    H_initial = torch.randn(1, num_nodes, node_dim)

    target_chi = 1.0 # Connected component
    target_mass = 12.0

    print("Training SheafNCA for Topological Reconstruction...")

    for epoch in range(100):
        optimizer.zero_grad()

        H_out = model(H_initial)

        grid_logits = torch.mean(H_out, dim=-1).view(1, H, W)
        grid = torch.sigmoid(grid_logits)

        # Loss 1: Euler Characteristic
        chi = relaxed_euler_torch(grid)
        loss_chi = (chi - target_chi)**2

        # Loss 2: Mass
        mass = torch.sum(grid)
        loss_mass = (mass - target_mass)**2

        # Loss 3: Spectral Gap (only for the first layer for efficiency)
        # Encouraging the learned sheaf structure to be connected
        first_layer = model.layers[0]
        gap = compute_sheaf_laplacian_spectral_gap(num_nodes, first_layer.edge_index, first_layer.W_maps, edge_dim, node_dim)
        loss_gap = torch.relu(0.5 - gap)

        loss = loss_chi + 0.1 * loss_mass + 1.0 * loss_gap

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Chi={chi.item():.4f}, Mass={mass.item():.4f}, Gap={gap.item():.4f}")

    # Final result
    with torch.no_grad():
        H_final = model(H_initial)
        grid_final = torch.sigmoid(torch.mean(H_final, dim=-1).view(H, W))
        binary_grid = (grid_final > 0.5).int()

        print("\nReconstructed Binary Grid:")
        print(binary_grid)

        # Compute exact chi
        V = torch.sum(binary_grid).item()
        E_h = torch.sum(binary_grid[:, :-1] * binary_grid[:, 1:]).item()
        E_v = torch.sum(binary_grid[:-1, :] * binary_grid[1:, :]).item()
        F = torch.sum(binary_grid[:-1, :-1] * binary_grid[:-1, 1:] * binary_grid[1:, :-1] * binary_grid[1:, 1:]).item()
        exact_chi = V - (E_h + E_v) + F
        print(f"Final Exact Chi: {exact_chi}")

if __name__ == "__main__":
    train_reconstruction()

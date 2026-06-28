import torch
import torch.optim as optim
from sheaf_nn import DeepSheafNetwork
from topo_torch import relaxed_euler_torch

def train():
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

    # Initialize Model
    model = DeepSheafNetwork(num_nodes, edges, node_dim, edge_dim, num_layers=num_layers, alpha=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Initial input features: Random
    H_initial = torch.randn(1, num_nodes, node_dim)

    target_chi = 1.0 # Target: A single connected component with no holes
    target_mass = 10.0

    print(f"Training to reach Target Chi: {target_chi}, Target Mass: {target_mass}")

    for epoch in range(200):
        optimizer.zero_grad()

        # Forward pass through Sheaf NN
        H_out = model(H_initial)

        # Interpret node features as probability grid
        # Mean across feature dimension, then sigmoid to [0, 1]
        grid_logits = torch.mean(H_out, dim=-1).view(1, H, W)
        grid = torch.sigmoid(grid_logits)

        # Topological Loss
        chi = relaxed_euler_torch(grid)
        loss_chi = (chi - target_chi)**2

        # Mass Loss
        mass = torch.sum(grid)
        loss_mass = (mass - target_mass)**2

        # Binarization penalty
        loss_bin = torch.sum(grid * (1.0 - grid))

        loss = loss_chi + 0.1 * loss_mass + 0.05 * loss_bin

        loss.backward()
        optimizer.step()

        # Optional: Project to Stiefel
        model.project_all_to_stiefel()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Chi={chi.item():.4f}, Mass={mass.item():.4f}")

    # Final result
    with torch.no_grad():
        H_final = model(H_initial)
        grid_final = torch.sigmoid(torch.mean(H_final, dim=-1).view(H, W))
        binary_grid = (grid_final > 0.5).int()

        print("\nFinal Binary Grid:")
        print(binary_grid)

        # Compute exact chi on binary grid
        V = torch.sum(binary_grid).item()
        # Horizontal edges
        E_h = torch.sum(binary_grid[:, :-1] * binary_grid[:, 1:]).item()
        # Vertical edges
        E_v = torch.sum(binary_grid[:-1, :] * binary_grid[1:, :]).item()
        # Faces
        F = torch.sum(binary_grid[:-1, :-1] * binary_grid[:-1, 1:] * binary_grid[1:, :-1] * binary_grid[1:, 1:]).item()

        exact_chi = V - (E_h + E_v) + F
        print(f"Final Exact Chi: {exact_chi}")
        print(f"Final Mass: {V}")

if __name__ == "__main__":
    train()

import torch
import torch.nn as nn
import torch.nn.functional as F

class SheafDiffusionLayer(nn.Module):
    def __init__(self, num_nodes, edges, node_dim, edge_dim, alpha=0.01):
        super(SheafDiffusionLayer, self).__init__()
        self.num_nodes = num_nodes
        self.num_edges = len(edges)
        self.d = node_dim
        self.de = edge_dim
        self.alpha = alpha

        self.W_maps = nn.Parameter(torch.randn(2 * self.num_edges, self.de, self.d))
        self.register_buffer('edge_index', torch.tensor(edges).t().contiguous())

    def load_from_manifold(self, loader):
        """
        Initializes W_maps using weights from the ManifoldLoader.
        """
        manifold_weights = loader.load_weights(2 * self.num_edges, edge_dim=self.de, node_dim=self.d)
        if manifold_weights.shape == self.W_maps.shape:
            self.W_maps.data.copy_(manifold_weights)
            print(f"Successfully loaded {2 * self.num_edges} manifold weights into W_maps.")
        else:
            print(f"Warning: Manifold weight shape {manifold_weights.shape} does not match W_maps shape {self.W_maps.shape}.")

    def forward(self, H):
        batch_size = H.size(0)
        W_src = self.W_maps[0::2]
        W_dst = self.W_maps[1::2]
        u_idx = self.edge_index[0]
        v_idx = self.edge_index[1]

        H_u = H[:, u_idx, :]
        H_v = H[:, v_idx, :]

        proj_u = torch.matmul(W_src, H_u.unsqueeze(-1)).squeeze(-1)
        proj_v = torch.matmul(W_dst, H_v.unsqueeze(-1)).squeeze(-1)

        Z = proj_u - proj_v

        grad_u = torch.matmul(W_src.transpose(-1, -2), Z.unsqueeze(-1)).squeeze(-1)
        grad_v = torch.matmul(W_dst.transpose(-1, -2), Z.unsqueeze(-1)).squeeze(-1)

        Delta_H = torch.zeros_like(H)
        expanded_u_idx = u_idx.view(1, -1, 1).expand(batch_size, -1, self.d)
        Delta_H.scatter_add_(1, expanded_u_idx, grad_u)
        expanded_v_idx = v_idx.view(1, -1, 1).expand(batch_size, -1, self.d)
        Delta_H.scatter_add_(1, expanded_v_idx, -grad_v)

        H_new = H - self.alpha * Delta_H
        return F.relu(H_new)

    def project_to_stiefel(self):
        with torch.no_grad():
            W = self.W_maps
            WtW = torch.matmul(W.transpose(-1, -2), W)
            e, v = torch.linalg.eigh(WtW)
            e_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(torch.clamp(e, min=1e-6)))
            WtW_inv_sqrt = torch.matmul(torch.matmul(v, e_inv_sqrt), v.transpose(-1, -2))
            W_new = torch.matmul(W, WtW_inv_sqrt)
            self.W_maps.copy_(W_new)

class SheafNCALayer(SheafDiffusionLayer):
    """
    Learned local update rule based on Sheaf residuals.
    Instead of simple diffusion, uses an MLP to compute the update.
    """
    def __init__(self, num_nodes, edges, node_dim, edge_dim, hidden_dim=16):
        super(SheafNCALayer, self).__init__(num_nodes, edges, node_dim, edge_dim)
        # MLP takes [local_feature, sheaf_residual]
        self.mlp = nn.Sequential(
            nn.Linear(node_dim + node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(self, H):
        batch_size = H.size(0)
        W_src = self.W_maps[0::2]
        W_dst = self.W_maps[1::2]
        u_idx = self.edge_index[0]
        v_idx = self.edge_index[1]

        H_u = H[:, u_idx, :]
        H_v = H[:, v_idx, :]

        proj_u = torch.matmul(W_src, H_u.unsqueeze(-1)).squeeze(-1)
        proj_v = torch.matmul(W_dst, H_v.unsqueeze(-1)).squeeze(-1)

        Z = proj_u - proj_v

        grad_u = torch.matmul(W_src.transpose(-1, -2), Z.unsqueeze(-1)).squeeze(-1)
        grad_v = torch.matmul(W_dst.transpose(-1, -2), Z.unsqueeze(-1)).squeeze(-1)

        Delta_H = torch.zeros_like(H)
        expanded_u_idx = u_idx.view(1, -1, 1).expand(batch_size, -1, self.d)
        Delta_H.scatter_add_(1, expanded_u_idx, grad_u)
        expanded_v_idx = v_idx.view(1, -1, 1).expand(batch_size, -1, self.d)
        Delta_H.scatter_add_(1, expanded_v_idx, -grad_v)

        # Local update: concatenate current feature and aggregated sheaf residual
        # concat_feat: [Batch, V, 2*d]
        concat_feat = torch.cat([H, Delta_H], dim=-1)
        update = self.mlp(concat_feat)

        H_new = H + update
        return F.relu(H_new)

class DeepSheafNetwork(nn.Module):
    def __init__(self, num_nodes, edges, node_dim, edge_dim, num_layers=3, alpha=0.01, layer_type='diffusion'):
        super(DeepSheafNetwork, self).__init__()
        if layer_type == 'diffusion':
            self.layers = nn.ModuleList([
                SheafDiffusionLayer(num_nodes, edges, node_dim, edge_dim, alpha=alpha)
                for _ in range(num_layers)
            ])
        elif layer_type == 'nca':
            self.layers = nn.ModuleList([
                SheafNCALayer(num_nodes, edges, node_dim, edge_dim)
                for _ in range(num_layers)
            ])

    def load_from_manifold(self, loader):
        """
        Initializes all layers using weights from the ManifoldLoader.
        """
        for i, layer in enumerate(self.layers):
            print(f"Loading manifold weights for layer {i}...")
            layer.load_from_manifold(loader)

    def forward(self, H):
        for layer in self.layers:
            H = layer(H)
        return H

    def project_all_to_stiefel(self):
        for layer in self.layers:
            layer.project_to_stiefel()

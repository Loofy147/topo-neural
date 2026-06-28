import torch
import torch.nn as nn
import torch.nn.functional as F

class SheafDiffusionLayer(nn.Module):
    """
    A single layer of Sheaf Diffusion/Convolution.
    """
    def __init__(self, num_nodes, edges, node_dim, edge_dim, alpha=0.01, use_stiefel=False):
        super(SheafDiffusionLayer, self).__init__()
        self.num_nodes = num_nodes
        self.num_edges = len(edges)
        self.d = node_dim
        self.de = edge_dim
        self.alpha = alpha
        self.use_stiefel = use_stiefel

        # Restriction maps: [2 * E, de, d]
        self.W_maps = nn.Parameter(torch.randn(2 * self.num_edges, self.de, self.d))
        self.register_buffer('edge_index', torch.tensor(edges).t().contiguous())

        # Optional: Projection to Stiefel manifold can be done after each optimizer step
        # or as part of the forward if we use a different parametrization.
        # For simplicity, we keep it as a method.

    def forward(self, H):
        batch_size = H.size(0)

        # 1. Edge Projection
        W_src = self.W_maps[0::2]
        W_dst = self.W_maps[1::2]
        u_idx = self.edge_index[0]
        v_idx = self.edge_index[1]

        H_u = H[:, u_idx, :]
        H_v = H[:, v_idx, :]

        proj_u = torch.matmul(W_src, H_u.unsqueeze(-1)).squeeze(-1)
        proj_v = torch.matmul(W_dst, H_v.unsqueeze(-1)).squeeze(-1)

        # 2. Coboundary Residual
        Z = proj_u - proj_v

        # 3. Gather-Reduction
        grad_u = torch.matmul(W_src.transpose(-1, -2), Z.unsqueeze(-1)).squeeze(-1)
        grad_v = torch.matmul(W_dst.transpose(-1, -2), Z.unsqueeze(-1)).squeeze(-1)

        Delta_H = torch.zeros_like(H)
        expanded_u_idx = u_idx.view(1, -1, 1).expand(batch_size, -1, self.d)
        Delta_H.scatter_add_(1, expanded_u_idx, grad_u)

        expanded_v_idx = v_idx.view(1, -1, 1).expand(batch_size, -1, self.d)
        Delta_H.scatter_add_(1, expanded_v_idx, -grad_v)

        # 4. Update
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

class DeepSheafNetwork(nn.Module):
    """
    A multi-layer Cellular Sheaf Neural Network.
    """
    def __init__(self, num_nodes, edges, node_dim, edge_dim, num_layers=3, alpha=0.01):
        super(DeepSheafNetwork, self).__init__()
        self.layers = nn.ModuleList([
            SheafDiffusionLayer(num_nodes, edges, node_dim, edge_dim, alpha=alpha)
            for _ in range(num_layers)
        ])

    def forward(self, H):
        for layer in self.layers:
            H = layer(H)
        return H

    def project_all_to_stiefel(self):
        for layer in self.layers:
            layer.project_to_stiefel()

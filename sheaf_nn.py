import torch
import torch.nn as nn
import torch.nn.functional as F

class CellularSheafNN(nn.Module):
    """
    Trainable Cellular Sheaf Neural Network (CSNN).
    Implements:
    1. Vertex and Edge Stalks.
    2. Learned Restriction Maps.
    3. Optimized Sheaf Diffusion/Convolution.
    4. Block-sparse Coboundary Multiplication.
    """
    def __init__(self, num_nodes, edges, node_dim, edge_dim, alpha=0.01):
        super(CellularSheafNN, self).__init__()
        self.num_nodes = num_nodes
        self.edges = edges # List of (u, v)
        self.num_edges = len(edges)
        self.d = node_dim
        self.de = edge_dim
        self.alpha = alpha

        # Restriction maps: For each edge, two maps (one for each vertex)
        # Shape: [2 * E, de, d]
        self.W_maps = nn.Parameter(torch.randn(2 * self.num_edges, self.de, self.d))

        # Initialize sources and destinations indices for scatter/gather
        self.register_buffer('edge_index', torch.tensor(edges).t().contiguous()) # [2, E]

    def forward(self, H):
        """
        H: Node Feature Tensor [Batch, V, d]
        Returns: Updated Node Feature Tensor [Batch, V, d]
        """
        batch_size = H.size(0)

        # 1. Edge Projection (Scatter-Src and Scatter-Dst)
        W_src = self.W_maps[0::2] # [E, de, d]
        W_dst = self.W_maps[1::2] # [E, de, d]

        u_idx = self.edge_index[0] # [E]
        v_idx = self.edge_index[1] # [E]

        H_u = H[:, u_idx, :] # [Batch, E, d]
        H_v = H[:, v_idx, :] # [Batch, E, d]

        # Projected features: [Batch, E, de]
        proj_u = torch.matmul(W_src, H_u.unsqueeze(-1)).squeeze(-1)
        proj_v = torch.matmul(W_dst, H_v.unsqueeze(-1)).squeeze(-1)

        # 2. Compute Edge Mismatch (Coboundary Residual Z)
        Z = proj_u - proj_v # [Batch, E, de]

        # 3. Gather-Reduction (Node Update)
        grad_u = torch.matmul(W_src.transpose(-1, -2), Z.unsqueeze(-1)).squeeze(-1) # [Batch, E, d]
        grad_v = torch.matmul(W_dst.transpose(-1, -2), Z.unsqueeze(-1)).squeeze(-1) # [Batch, E, d]

        # Vectorized Aggregate back to nodes
        Delta_H = torch.zeros_like(H)

        expanded_u_idx = u_idx.view(1, -1, 1).expand(batch_size, -1, self.d)
        Delta_H.scatter_add_(1, expanded_u_idx, grad_u)

        expanded_v_idx = v_idx.view(1, -1, 1).expand(batch_size, -1, self.d)
        Delta_H.scatter_add_(1, expanded_v_idx, -grad_v)

        # 4. Update H: H^(l+1) = sigma(H^(l) - alpha * Delta_F * H)
        H_new = H - self.alpha * Delta_H
        return F.relu(H_new)

    def project_to_stiefel(self):
        """
        Constrains restriction maps to the Stiefel Manifold (orthogonal matrices).
        W_projected = W * (W^T * W)^(-1/2)
        """
        with torch.no_grad():
            W = self.W_maps # [2*E, de, d]
            # Compute W^T * W
            WtW = torch.matmul(W.transpose(-1, -2), W) # [2*E, d, d]

            # Eigen decomposition
            e, v = torch.linalg.eigh(WtW)
            # Inverse square root of eigenvalues
            e_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(torch.clamp(e, min=1e-6)))

            WtW_inv_sqrt = torch.matmul(torch.matmul(v, e_inv_sqrt), v.transpose(-1, -2))

            W_new = torch.matmul(W, WtW_inv_sqrt)
            self.W_maps.copy_(W_new)

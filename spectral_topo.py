import torch

def assemble_coboundary_matrix(num_nodes, edge_index, W_maps, de, d):
    """
    Assembles the Coboundary Matrix D_F as a dense tensor.
    Vectorized implementation.
    W_maps: [2 * E, de, d]
    edge_index: [2, E]
    Returns: D_F of shape [E * de, V * d]
    """
    num_edges = edge_index.size(1)
    device = W_maps.device
    dtype = W_maps.dtype

    D_F = torch.zeros(num_edges * de, num_nodes * d, device=device, dtype=dtype)

    # Indices for row-wise blocks
    row_starts = torch.arange(num_edges, device=device) * de

    # Source blocks (W_u)
    u_idx = edge_index[0] # [E]
    u_col_starts = u_idx * d # [E]

    # Destination blocks (-W_v)
    v_idx = edge_index[1] # [E]
    v_col_starts = v_idx * d # [E]

    W_src = W_maps[0::2] # [E, de, d]
    W_dst = W_maps[1::2] # [E, de, d]

    # Create grid of offsets within each block
    ii, jj = torch.meshgrid(torch.arange(de, device=device), torch.arange(d, device=device), indexing='ij')

    # Expand to all edges
    row_indices = (row_starts.view(-1, 1, 1) + ii.view(1, de, d)).view(-1)
    col_indices_src = (u_col_starts.view(-1, 1, 1) + jj.view(1, de, d)).view(-1)
    col_indices_dst = (v_col_starts.view(-1, 1, 1) + jj.view(1, de, d)).view(-1)

    # Flattened indices for 1D scatter
    stride = num_nodes * d
    idx_src = row_indices * stride + col_indices_src
    idx_dst = row_indices * stride + col_indices_dst

    D_F.view(-1).scatter_add_(0, idx_src, W_src.reshape(-1))
    D_F.view(-1).scatter_add_(0, idx_dst, -W_dst.reshape(-1))

    return D_F

def compute_sheaf_laplacian_spectral_gap(num_nodes, edge_index, W_maps, de, d):
    """
    Computes the second smallest eigenvalue of the Sheaf Laplacian.
    This serves as a differentiable proxy for connectivity/alignment.
    """
    D_F = assemble_coboundary_matrix(num_nodes, edge_index, W_maps, de, d)
    L_F = torch.matmul(D_F.t(), D_F) # [V*d, V*d]

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(L_F)

    # Return the second smallest eigenvalue
    if eigenvalues.numel() > 1:
        return eigenvalues[1]
    else:
        return eigenvalues[0]

def spectral_connectivity_loss(num_nodes, edge_index, W_maps, de, d, target_gap=0.1):
    """
    Loss that encourages the Sheaf Laplacian to have a spectral gap.
    """
    gap = compute_sheaf_laplacian_spectral_gap(num_nodes, edge_index, W_maps, de, d)
    return torch.relu(target_gap - gap)

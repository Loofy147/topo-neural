import torch

def assemble_coboundary_matrix(num_nodes, edge_index, W_maps, de, d):
    """
    Assembles the Coboundary Matrix D_F as a dense tensor.
    W_maps: [2 * E, de, d]
    edge_index: [2, E]
    Returns: D_F of shape [E * de, V * d]
    """
    num_edges = edge_index.size(1)
    D_F = torch.zeros(num_edges * de, num_nodes * d, device=W_maps.device, dtype=W_maps.dtype)

    for e in range(num_edges):
        u = edge_index[0, e]
        v = edge_index[1, e]

        W_u = W_maps[2 * e]     # [de, d]
        W_v = W_maps[2 * e + 1] # [de, d]

        # Place W_u and -W_v in the coboundary matrix
        D_F[e * de : (e + 1) * de, u * d : (u + 1) * d] = W_u
        D_F[e * de : (e + 1) * de, v * d : (v + 1) * d] = -W_v

    return D_F

def compute_sheaf_laplacian_spectral_gap(num_nodes, edge_index, W_maps, de, d):
    """
    Computes the second smallest eigenvalue of the Sheaf Laplacian.
    This serves as a differentiable proxy for connectivity/alignment.
    """
    D_F = assemble_coboundary_matrix(num_nodes, edge_index, W_maps, de, d)
    L_F = torch.matmul(D_F.t(), D_F) # [V*d, V*d]

    # Compute eigenvalues
    # eigh is for symmetric matrices
    eigenvalues = torch.linalg.eigvalsh(L_F)

    # For a standard connected graph Laplacian, the first eigenvalue is 0.
    # For a Sheaf Laplacian, the number of 0 eigenvalues corresponds to the
    # dimension of the space of global sections (H^0).
    # We return the second smallest eigenvalue as a measure of "spectral gap".
    if eigenvalues.numel() > 1:
        return eigenvalues[1]
    else:
        return eigenvalues[0]

def spectral_connectivity_loss(num_nodes, edge_index, W_maps, de, d, target_gap=0.1):
    """
    Loss that encourages the Sheaf Laplacian to have a spectral gap.
    """
    gap = compute_sheaf_laplacian_spectral_gap(num_nodes, edge_index, W_maps, de, d)
    # We want gap to be at least target_gap
    return torch.relu(target_gap - gap)

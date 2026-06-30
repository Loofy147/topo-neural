import torch

def relaxed_euler_torch(grid):
    """
    Continuous relaxation of the Euler Characteristic in PyTorch.
    Supports backpropagation.
    grid: Tensor of shape [Batch, H, W] with values in [0, 1]
    """
    # V: Sum of vertices
    V = torch.sum(grid, dim=(1, 2))

    # E_h: Horizontal edges
    E_h = torch.sum(grid[:, :, :-1] * grid[:, :, 1:], dim=(1, 2))

    # E_v: Vertical edges
    E_v = torch.sum(grid[:, :-1, :] * grid[:, 1:, :], dim=(1, 2))

    E = E_h + E_v

    # F: Faces (quads)
    F = torch.sum(grid[:, :-1, :-1] * grid[:, :-1, 1:] * grid[:, 1:, :-1] * grid[:, 1:, 1:], dim=(1, 2))

    return V - E + F

def compute_euler_binary_torch(grid_binary):
    """
    Exact Euler Characteristic for binary tensors in PyTorch.
    grid_binary: Tensor of shape [Batch, H, W] with values in {0, 1}
    """
    V = torch.sum(grid_binary, dim=(1, 2))

    # Use logical_and for exactness, though multiplication works for binary
    E_h = torch.sum(grid_binary[:, :, :-1] * grid_binary[:, :, 1:], dim=(1, 2))
    E_v = torch.sum(grid_binary[:, :-1, :] * grid_binary[:, 1:, :], dim=(1, 2))
    E = E_h + E_v

    F = torch.sum(grid_binary[:, :-1, :-1] * grid_binary[:, :-1, 1:] * grid_binary[:, 1:, :-1] * grid_binary[:, 1:, 1:], dim=(1, 2))

    return V - E + F

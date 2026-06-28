# Unified Topology NCA

This repository implements a unified, optimized framework for a **Topological Neural Network (TNN)** as described in the blueprint. It integrates verified optimizations for shape completion, generative alignment, and spatial pattern propagation on discrete 2D grid complexes.

## Features

1.  **Fast Kronecker-assembled Sheaf Laplacians**: Efficiently analyzes connectivity using sparse Kronecker formulations.
2.  **Bitwise Euler Characteristic Tracking**: Uses boolean logic and shared-slice vertical overlap for high-speed topological invariant calculation.
3.  **Mass-Regulated Shape Optimization**: Guides the evolution of topological shapes using a composite objective function that prevents trivial collapses.

## Getting Started

### Prerequisites

-   Python 3.x
-   NumPy
-   SciPy

### Installation

```bash
pip install numpy scipy
```

### Usage

The core logic is contained within the `UnifiedTopologyNCA` class in `topo_nca.py`.

```python
from topo_nca import UnifiedTopologyNCA

# Initialize on a 6x6 grid
nca = UnifiedTopologyNCA(6, 6)

# Generate a shape with target Euler Characteristic (chi) and target mass
binary_grid, chi, mass = nca.train_shape(target_chi=0, target_mass=12, steps=1000)
```

### Running the Test Script

A verification script is provided in `main.py`. Execute it to see the framework in action:

```bash
python main.py
```

## Architecture

-   **Module A (The Guard)**: Fast Bitwise Topological Invariants.
-   **Module B (The Router)**: Sparse Kronecker Sheaf Laplacian.
-   **Module C (The Engine)**: Mass-Regulated Generative Loop.

## Trainable Cellular Sheaf Neural Network (CSNN)

This module implements a Trainable Cellular Sheaf Neural Network using PyTorch.

### Features
- **Vertex & Edge Stalks**: High-dimensional representations for nodes and edges.
- **Learned Restriction Maps**: Parameterized linear maps defining connections.
- **Optimized Sheaf Diffusion**: Vectorized forward pass using scatter-gather operations.
- **Stiefel Manifold Projection**: Optional constraint to ensure orthogonal restriction maps for training stability.

### Usage

```python
from sheaf_nn import CellularSheafNN
import torch

num_nodes = 3
edges = [(0, 1), (1, 2), (2, 0)]
model = CellularSheafNN(num_nodes, edges, node_dim=4, edge_dim=2)

H = torch.randn(1, num_nodes, 4)
H_new = model(H)
```

### Running Tests

```bash
python test_sheaf_nn.py
```

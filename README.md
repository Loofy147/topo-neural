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

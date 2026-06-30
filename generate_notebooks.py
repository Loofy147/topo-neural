import nbformat as nbf
import os

def read_file_content(path):
    with open(path, 'r') as f:
        return f.read()

def create_training_notebook():
    nb = nbf.v4.new_notebook()

    nb.cells.append(nbf.v4.new_markdown_cell("# Unified Topology NCA: High-Leverage Training\nThis notebook implements the full training pipeline with GPU support, Kaggle Hub integration, and Topological Regularizers."))

    nb.cells.append(nbf.v4.new_code_cell("!pip install kagglehub wandb"))

    # Bundle necessary scripts
    scripts = [
        'monitoring_utils.py',
        'data_utils.py',
        'kaggle_utils.py',
        'kaggle_hub_manager.py',
        'weights_loader.py',
        'sheaf_nn.py',
        'topo_torch.py',
        'spectral_topo.py',
        'train_high_leverage.py'
    ]

    for script in scripts:
        content = read_file_content(script)
        nb.cells.append(nbf.v4.new_code_cell(f"%%writefile {script}\n{content}"))

    code_run = """
import torch
import os
from train_high_leverage import train

# Set your Kaggle Model Handle if you want to push to Hub
# os.environ['KAGGLE_MODEL_HANDLE'] = 'username/model/pytorch/version'
# os.environ['WANDB_API_KEY'] = 'your_key'
# os.environ['KAGGLE_API_TOKEN'] = '...' # If needed inside the notebook

# Ensure directories exist
os.makedirs('kaggle_data/stratos_manifold', exist_ok=True)
os.makedirs('kaggle_data/stratoscot', exist_ok=True)
os.makedirs('kaggle_data/omega_manifold', exist_ok=True)
os.makedirs('kaggle_data/fso_manifold', exist_ok=True)
os.makedirs('kaggle_data/precision_data', exist_ok=True)

# Run training
train(dry_run=False)
"""
    nb.cells.append(nbf.v4.new_code_cell(code_run.strip()))

    with open('Training_Full_Pipeline.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_assessment_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell("# Assessment and Improvement\nAnalyze training results and discover new manifold resources."))
    # ... (keeping it simple as before or bundling if needed)
    with open('Assessment_and_Improvement.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    create_training_notebook()
    create_assessment_notebook()

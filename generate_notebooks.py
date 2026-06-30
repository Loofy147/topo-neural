import nbformat as nbf
import os

def read_file_content(path):
    with open(path, 'r') as f:
        return f.read()

def add_metadata(nb):
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }
    nb.metadata.language_info = {
        "codemirror_mode": {
            "name": "ipython",
            "version": 3
        },
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.10.12"
    }

def create_training_notebook():
    nb = nbf.v4.new_notebook()
    add_metadata(nb)

    nb.cells.append(nbf.v4.new_markdown_cell("# Unified Topology NCA: High-Leverage Training\nTraining with GPU support and Topological Regularizers."))

    # Environment fix via bash to avoid pre-importing torch
    nb.cells.append(nbf.v4.new_code_cell("""
%%bash
echo "Installing compatible environment..."
# Install torch 2.0.0 which supports sm_60 (Tesla P100)
pip install --no-cache-dir --force-reinstall torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install --upgrade kaggle kagglehub wandb
""".strip()))

    nb.cells.append(nbf.v4.new_code_cell("""
import torch
import os

print(f"Torch version: {torch.__version__}")
if torch.cuda.is_available():
    prop = torch.cuda.get_device_properties(0)
    print(f"Device: {prop.name}, Compute Capability: {prop.major}.{prop.minor}")
    try:
        x = torch.randn(1, device='cuda')
        print("CUDA Test Successful!")
    except Exception as e:
        print(f"CUDA Test Failed: {e}. Check if the torch build supports sm_{prop.major}{prop.minor}")
else:
    print("CUDA not available.")
""".strip()))

    # Environment Setup and Dataset Symlinking
    nb.cells.append(nbf.v4.new_code_cell("""
import os
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_453cfb028676f79df571e5b2a8ee6afd'

# Symlink datasets
os.makedirs('kaggle_data', exist_ok=True)

input_dir = '/kaggle/input'
mapping = {
    'stratos-manifold-v4': 'stratos_manifold',
    'stratoscot': 'stratoscot',
    'stratos-omega-manifold-v3': 'omega_manifold',
    'fso-manifold': 'fso_manifold',
    'precision-system-v3-data': 'precision_data'
}

if os.path.exists(input_dir):
    for root, dirs, files in os.walk(input_dir):
        item = os.path.basename(root)
        if item in mapping:
            dst = os.path.join('kaggle_data', mapping[item])
            if not os.path.exists(dst):
                # If it's a nested directory structure, link the child if it contains data
                sub = os.listdir(root)
                target_src = root
                if len(sub) == 1 and os.path.isdir(os.path.join(root, sub[0])):
                    target_src = os.path.join(root, sub[0])

                os.symlink(target_src, dst)
                print(f"Linked {target_src} -> {dst}")
""".strip()))

    # Bundle scripts
    scripts = ['monitoring_utils.py', 'data_utils.py', 'kaggle_utils.py', 'kaggle_hub_manager.py',
               'weights_loader.py', 'sheaf_nn.py', 'topo_torch.py', 'spectral_topo.py', 'train_high_leverage.py']

    for script in scripts:
        content = read_file_content(script)
        nb.cells.append(nbf.v4.new_code_cell(f"%%writefile {script}\n{content}"))

    nb.cells.append(nbf.v4.new_code_cell("""
from train_high_leverage import train
train(dry_run=False)
""".strip()))

    with open('Training_Full_Pipeline.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_assessment_notebook():
    nb = nbf.v4.new_notebook()
    add_metadata(nb)
    nb.cells.append(nbf.v4.new_markdown_cell("# Assessment and Improvement"))
    with open('Assessment_and_Improvement.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    create_training_notebook()
    create_assessment_notebook()

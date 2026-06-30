import nbformat as nbf
import os

def read_file_content(path):
    with open(path, 'r') as f:
        return f.read()

def add_metadata(nb):
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py", "mimetype": "text/x-python", "name": "python",
        "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.10.12"
    }

def create_training_notebook():
    nb = nbf.v4.new_notebook()
    add_metadata(nb)

    nb.cells.append(nbf.v4.new_markdown_cell("# Unified Topology NCA: Ultra-Long 100k Epoch Training\nHigh-leverage training suite with sm_60 compatibility and automated checkpointing."))

    nb.cells.append(nbf.v4.new_code_cell("""
%%bash
echo "Fixing environment..."
pip install --no-cache-dir --force-reinstall torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install --upgrade kaggle kagglehub wandb
""".strip()))

    nb.cells.append(nbf.v4.new_code_cell("""
import torch
import os
if torch.cuda.is_available():
    prop = torch.cuda.get_device_properties(0)
    print(f"Device: {prop.name}, Compute Capability: {prop.major}.{prop.minor}")
    try:
        x = torch.randn(1, device='cuda'); print("CUDA Test Successful!")
    except Exception as e:
        print(f"CUDA Test Failed: {e}. Build might not support sm_{prop.major}{prop.minor}")
else: print("CUDA not available.")
""".strip()))

    nb.cells.append(nbf.v4.new_code_cell("""
import os
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_453cfb028676f79df571e5b2a8ee6afd'
os.makedirs('kaggle_data', exist_ok=True)
input_dir = '/kaggle/input'
mapping = {'stratos-manifold-v4': 'stratos_manifold', 'stratoscot': 'stratoscot', 'stratos-omega-manifold-v3': 'omega_manifold', 'fso-manifold': 'fso_manifold', 'precision-system-v3-data': 'precision_data'}
if os.path.exists(input_dir):
    for root, dirs, files in os.walk(input_dir):
        item = os.path.basename(root)
        if item in mapping:
            dst = os.path.join('kaggle_data', mapping[item])
            if not os.path.exists(dst):
                sub = os.listdir(root)
                target_src = os.path.join(root, sub[0]) if len(sub) == 1 and os.path.isdir(os.path.join(root, sub[0])) else root
                os.symlink(target_src, dst); print(f"Linked {target_src} -> {dst}")
""".strip()))

    scripts = ['monitoring_utils.py', 'data_utils.py', 'kaggle_utils.py', 'kaggle_hub_manager.py', 'weights_loader.py', 'sheaf_nn.py', 'topo_torch.py', 'spectral_topo.py', 'train_high_leverage.py']
    for script in scripts:
        nb.cells.append(nbf.v4.new_code_cell(f"%%writefile {script}\n{read_file_content(script)}"))

    nb.cells.append(nbf.v4.new_code_cell("""
from train_high_leverage import train
# Starting the 100,000 epoch run
train(dry_run=False, num_epochs=100000, checkpoint_freq=500)
""".strip()))

    with open('Training_Full_Pipeline.ipynb', 'w') as f: nbf.write(nb, f)

def create_assessment_notebook():
    nb = nbf.v4.new_notebook(); add_metadata(nb)
    nb.cells.append(nbf.v4.new_markdown_cell("# Assessment and Improvement"))
    with open('Assessment_and_Improvement.ipynb', 'w') as f: nbf.write(nb, f)

if __name__ == '__main__':
    create_training_notebook(); create_assessment_notebook()

import nbformat as nbf
import os

def read_file_content(path):
    if not os.path.exists(path):
        return f"# File {path} not found."
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

    nb.cells.append(nbf.v4.new_markdown_cell("# Unified Topology NCA & ARC Manifold Reasoning v4\nEnhanced autonomous reasoning with Neural Agents and CGCE."))

    nb.cells.append(nbf.v4.new_code_cell("""
%%bash
echo "Fixing environment..."
pip install --no-cache-dir --force-reinstall torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install --upgrade kaggle kagglehub wandb pandas numpy
""".strip()))

    nb.cells.append(nbf.v4.new_code_cell("""
import os
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_453cfb028676f79df571e5b2a8ee6afd'
os.environ['KAGGLE_USERNAME'] = 'hichambedrani'
os.environ['KAGGLE_KEY'] = 'KGAT_453cfb028676f79df571e5b2a8ee6afd'

os.makedirs('kaggle_data', exist_ok=True)
!mkdir -p kaggle_data/sec_financials
!kaggle datasets download -d securities-exchange-commission/financial-statement-extracts -p kaggle_data/sec_financials --unzip
""".strip()))

    scripts = [
        'monitoring_utils.py', 'data_utils.py', 'kaggle_utils.py', 'kaggle_hub_manager.py',
        'weights_loader.py', 'sheaf_nn.py', 'topo_torch.py', 'spectral_topo.py',
        'train_high_leverage.py', 'cgce.py', 'autonomous_manifold_v4.py', 'hybrid_solver.py',
        'financial_data_utils.py', 'extract_financials.py', 'populate_financial_manifold.py',
        'train_financial_leverage.py', 'arc_manifold_solver.py'
    ]
    for script in scripts:
        nb.cells.append(nbf.v4.new_code_cell(f"%%writefile {script}\n{read_file_content(script)}"))

    nb.cells.append(nbf.v4.new_markdown_cell("## Population of Financial Manifold"))
    nb.cells.append(nbf.v4.new_code_cell("!python3 extract_financials.py\n!python3 populate_financial_manifold.py"))

    nb.cells.append(nbf.v4.new_markdown_cell("## ARC Manifold Reasoning v4"))
    nb.cells.append(nbf.v4.new_code_cell("""
from arc_manifold_solver import ARCManifoldSolverV3
import json

sample_task = {
    "train": [{"input": [[1, 1], [0, 0]], "output": [[0, 0], [1, 1]]}],
    "test": [{"input": [[1, 0], [1, 0]]}]
}
solver = ARCManifoldSolverV3()
prediction = solver.solve_task("sample_v4", sample_task)
print(f"V4 Prediction: {json.dumps(prediction, indent=2)}")
""".strip()))

    with open('Training_Full_Pipeline.ipynb', 'w') as f: nbf.write(nb, f)

def create_assessment_notebook():
    nb = nbf.v4.new_notebook(); add_metadata(nb)
    nb.cells.append(nbf.v4.new_markdown_cell("# Assessment and Improvement"))
    nb.cells.append(nbf.v4.new_code_cell("!python3 arc_manifold_solver.py"))
    with open('Assessment_and_Improvement.ipynb', 'w') as f: nbf.write(nb, f)

if __name__ == '__main__':
    create_training_notebook(); create_assessment_notebook()

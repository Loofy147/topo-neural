import torch, torch.nn as nn, torch.optim as optim, os
from sheaf_nn import DeepSheafNetwork
from weights_loader import ManifoldLoader
from financial_data_utils import get_financial_dataloader
def train(dry_run=False, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepSheafNetwork(8, [(i, (i+1)%8) for i in range(8)], 32, 32, num_layers=8, layer_type='nca').to(device)
    try: model.load_from_manifold(ManifoldLoader(directory='kaggle_data/financial_manifold'))
    except: pass
    head = nn.Sequential(nn.Linear(32, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 8)).to(device)
    opt = optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-4)
    crit = nn.BCEWithLogitsLoss()
    dl = get_financial_dataloader('extracted_assets_liabilities.csv', batch_size=64)
    for epoch in range(num_epochs if not dry_run else 1):
        model.train()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(head(model(x.unsqueeze(-1).repeat(1, 1, 32))).mean(1), y).backward()
            opt.step()
        print(f"Epoch {epoch} complete")
if __name__ == "__main__": train(dry_run=True)

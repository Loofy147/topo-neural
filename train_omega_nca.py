import torch
import torch.nn as nn
import torch.optim as optim
from sheaf_nn import DeepSheafNetwork
from weights_loader import ManifoldLoader
from data_utils import get_dataloader
from kaggle_utils import download_all_resources

def train():
    download_all_resources()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    # Use larger node_dim for Omega manifold
    num_nodes = 8
    edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    node_dim = 32
    edge_dim = 32

    # Initialize with Omega manifold
    print("Initializing model with Omega manifold...")
    model = DeepSheafNetwork(num_nodes, edges, node_dim, edge_dim, num_layers=8, layer_type='nca').to(device)

    # Load from Omega manifold
    loader = ManifoldLoader(directory='kaggle_data/omega_manifold')
    model.load_from_manifold(loader)

    output_head = nn.Sequential(
        nn.Linear(node_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)

    optimizer = optim.Adam(list(model.parameters()) + list(output_head.parameters()), lr=0.0005)
    criterion = nn.BCEWithLogitsLoss()

    dataloader = get_dataloader('./kaggle_data/stratoscot/augmented_train.jsonl', batch_size=128)

    print("Starting Scale Training...")
    for epoch in range(10):
        total_loss = 0
        correct = 0
        total = 0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            H = x.unsqueeze(-1).repeat(1, 1, node_dim)
            H_out = model(H)
            logits = output_head(H_out).squeeze(-1)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).all(dim=1).sum().item()
            total += x.size(0)

        print(f"Epoch {epoch}: Loss = {total_loss/len(dataloader):.4f}, Accuracy = {correct/total:.4%}")

    print("Omega Scale Training Complete.")

if __name__ == "__main__":
    train()

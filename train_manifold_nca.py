import torch
import torch.nn as nn
import torch.optim as optim
from sheaf_nn import DeepSheafNetwork
from weights_loader import ManifoldLoader
from data_utils import get_dataloader

def bit_error_rate(logits, targets):
    preds = (torch.sigmoid(logits) > 0.5).float()
    return torch.mean((preds != targets).float())

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    # Dataset
    data_path = './kaggle_data/stratoscot/augmented_train.jsonl'
    dataloader = get_dataloader(data_path, batch_size=64)

    # Model configuration for 8-bit input
    num_nodes = 8
    edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]

    node_dim = 32
    edge_dim = 32

    print("Initializing model with manifold weights...")
    model = DeepSheafNetwork(num_nodes, edges, node_dim, edge_dim, num_layers=5, layer_type='nca').to(device)

    loader = ManifoldLoader()
    model.load_from_manifold(loader)

    output_head = nn.Sequential(
        nn.Linear(node_dim, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    ).to(device)

    optimizer = optim.Adam(list(model.parameters()) + list(output_head.parameters()), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    print("Starting training loop...")
    for epoch in range(10):
        total_loss = 0
        total_ber = 0
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
            total_ber += bit_error_rate(logits, y).item()

            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).all(dim=1).sum().item()
            total += x.size(0)

        avg_loss = total_loss / len(dataloader)
        avg_ber = total_ber / len(dataloader)
        accuracy = correct / total
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, BER = {avg_ber:.4f}, Accuracy = {accuracy:.4%}")

    print("Training complete.")

if __name__ == "__main__":
    train()

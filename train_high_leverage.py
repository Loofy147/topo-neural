import torch
import torch.nn as nn
import torch.optim as optim
from sheaf_nn import DeepSheafNetwork
from weights_loader import ManifoldLoader
from data_utils import get_dataloader
from monitoring_utils import calculate_leverage, monitor_model_health

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Large Scale High-Leverage Training ---")
    print(f"Device: {device}")

    # Configuration for deep network
    num_nodes = 8
    edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    # Adding more connectivity for "high leverage"
    for i in range(num_nodes):
        edges.append((i, (i + 2) % num_nodes))

    node_dim = 32
    edge_dim = 32
    num_layers = 12 # Deep network

    model = DeepSheafNetwork(num_nodes, edges, node_dim, edge_dim, num_layers=num_layers, layer_type='nca').to(device)

    # Cross-manifold initialization
    print("Initializing from Multiple Manifolds...")
    stratos_loader = ManifoldLoader(source='stratos')
    omega_loader = ManifoldLoader(source='omega')

    for i, layer in enumerate(model.layers):
        loader = stratos_loader if i % 2 == 0 else omega_loader
        layer.load_from_manifold(loader)

    output_head = nn.Sequential(
        nn.Linear(node_dim, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Linear(128, 1)
    ).to(device)

    optimizer = optim.AdamW(list(model.parameters()) + list(output_head.parameters()), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.BCEWithLogitsLoss()

    dataloader = get_dataloader('./kaggle_data/stratoscot/augmented_train.jsonl', batch_size=256)

    print("Starting Epochs...")
    for epoch in range(10):
        model.train()
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

            # Monitoring
            grad_norm = monitor_model_health(model)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).all(dim=1).sum().item()
            total += x.size(0)
            total_ber += torch.mean((preds != y).float()).item()

        scheduler.step()

        avg_acc = correct / total
        avg_ber = total_ber / len(dataloader)

        # Calculate leverage
        with torch.no_grad():
            w_norm = sum(p.norm(2).item() for p in model.parameters())
            leverage = calculate_leverage(avg_acc, avg_ber, w_norm)

        print(f"Epoch {epoch}: Loss={total_loss/len(dataloader):.4f}, Acc={avg_acc:.2%}, BER={avg_ber:.4f}, Leverage={leverage:.4f}")

    print("High-Leverage Model Training Complete.")

if __name__ == "__main__":
    train()

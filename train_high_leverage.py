import torch
import torch.nn as nn
import torch.optim as optim
from sheaf_nn import DeepSheafNetwork
from weights_loader import ManifoldLoader
from data_utils import get_dataloader
from monitoring_utils import calculate_leverage, monitor_model_health, WandbLogger
from kaggle_hub_manager import save_and_push_to_hub
import json
import os

def train(dry_run=False):
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

    config = {
        "num_nodes": num_nodes,
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "num_layers": num_layers,
        "lr": 0.0001,
        "weight_decay": 0.01
    }

    model = DeepSheafNetwork(num_nodes, edges, node_dim, edge_dim, num_layers=num_layers, layer_type='nca').to(device)

    # Initialize wandb
    logger = WandbLogger(project_name="topo-neural-high-leverage", config=config)

    # Mixed Precision Scaling
    if device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = torch.amp.GradScaler('cpu', enabled=False)

    # Cross-manifold initialization
    print("Initializing from Multiple Manifolds...")
    try:
        stratos_loader = ManifoldLoader(source='stratos')
        omega_loader = ManifoldLoader(source='omega')

        for i, layer in enumerate(model.layers):
            loader = stratos_loader if i % 2 == 0 else omega_loader
            layer.load_from_manifold(loader)
    except Exception as e:
        print(f"Initialization from manifold failed: {e}. Using random weights.")

    output_head = nn.Sequential(
        nn.Linear(node_dim, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Linear(128, 8) # Outputting 8 bits
    ).to(device)

    optimizer = optim.AdamW(list(model.parameters()) + list(output_head.parameters()), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.BCEWithLogitsLoss()

    report_file = 'TRAINING_REPORT.jsonl'

    # Try to get dataloader
    try:
        dataloader = get_dataloader('./kaggle_data/stratoscot/augmented_train.jsonl', batch_size=256)
    except Exception as e:
        print(f"Dataloader failed: {e}. Using dummy data for testing.")
        x_dummy = torch.randn(10, 8)
        y_dummy = torch.randint(0, 2, (10, 8)).float()
        dataloader = [(x_dummy, y_dummy)]

    num_epochs = 10 if not dry_run else 1
    best_leverage = -1

    print("Starting Epochs...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_ber = 0
        correct = 0
        total = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'
            with torch.amp.autocast(autocast_device, enabled=(device.type == 'cuda')):
                H = x.unsqueeze(-1).repeat(1, 1, node_dim)
                H_out = model(H)
                logits = output_head(H_out).mean(dim=1)

                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            # Monitoring
            grad_norm = monitor_model_health(model)
            if device.type == 'cuda':
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).all(dim=1).sum().item()
            total += x.size(0)
            total_ber += torch.mean((preds != y).float()).item()

            if dry_run and batch_idx >= 2:
                break

        scheduler.step()

        avg_acc = correct / total if total > 0 else 0
        avg_ber = total_ber / len(dataloader)

        # Calculate leverage
        with torch.no_grad():
            w_norm = sum(p.norm(2).item() for p in model.parameters())
            leverage = calculate_leverage(avg_acc, avg_ber, w_norm)

        metrics = {
            "epoch": epoch,
            "loss": total_loss / len(dataloader),
            "accuracy": avg_acc,
            "ber": avg_ber,
            "leverage": leverage,
            "grad_norm": grad_norm
        }

        logger.log(metrics)

        with open(report_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

        print(f"Epoch {epoch}: Loss={metrics['loss']:.4f}, Acc={avg_acc:.2%}, BER={avg_ber:.4f}, Leverage={leverage:.4f}")

        if leverage > best_leverage and not dry_run:
            best_leverage = leverage
            handle = os.environ.get('KAGGLE_MODEL_HANDLE')
            if handle:
                save_and_push_to_hub(model, optimizer, epoch, metrics, handle)

    logger.finish()
    print("High-Leverage Model Training Complete.")

if __name__ == "__main__":
    train(dry_run=True)

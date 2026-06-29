import torch
import numpy as np

def calculate_leverage(accuracy, ber, weights_norm):
    """
    Custom 'Higher Leverage' metric that balances performance and parameter efficiency.
    Inspired by the Precision Targeting Engine.
    """
    # accuracy is [0, 1], ber is [0, 1]
    # We want high accuracy, low BER, and stable weight norms.
    efficiency = 1.0 / (1.0 + weights_norm)
    leverage = (accuracy * (1.0 - ber) * efficiency) * 100
    return leverage

def monitor_model_health(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

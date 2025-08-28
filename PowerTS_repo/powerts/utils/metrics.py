
import torch

def mse(pred, true):
    return torch.mean((pred - true) ** 2).item()

def mae(pred, true):
    return torch.mean(torch.abs(pred - true)).item()

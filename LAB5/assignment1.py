import torch
from classification import TransformerClassifier
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def train(model, loader, opt, loss_fn, device):
    model.train()
    total = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)

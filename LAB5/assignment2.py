import torch
from sequential_labeling import TransformerNER

def train(model, loader, opt, loss_fn):
    model.train()
    for x,y in loader:
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        opt.step()

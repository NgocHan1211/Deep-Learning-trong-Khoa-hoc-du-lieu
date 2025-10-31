import torch
from torch import nn
from torch.nn import functional as F

class Perceptron_1_layer(nn.Module):
    def __init__(self, image_size=(28,28), num_labels=10):
        super().__init__()
        w, h = image_size
        self.linear = nn.Linear(in_features=w*h, out_features=num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        x = x.reshape(bs, -1)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)

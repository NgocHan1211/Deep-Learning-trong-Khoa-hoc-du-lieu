import torch
from torch import nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.h = heads

        self.qkv = nn.Linear(d_model, d_model*3)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.h, self.d_k).transpose(1,2) for t in qkv]

        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        att = torch.softmax(scores, dim=-1)
        out = (att @ v).transpose(1,2).contiguous().view(B,T,C)
        return self.fc(out)

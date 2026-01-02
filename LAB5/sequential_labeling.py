import torch
from torch import nn
from transformer import TransformerEncoder, PositionalEncoding

class TransformerNER(nn.Module):
    def __init__(self, vocab_size, n_tags, d_model=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model)
        self.encoder = TransformerEncoder(3, d_model, 4, 256, 0.1)
        self.fc = nn.Linear(d_model, n_tags)

    def forward(self, x):
        mask = (x != 0).unsqueeze(1).unsqueeze(2)
        x = self.pe(self.embed(x))
        x = self.encoder(x, mask)
        return self.fc(x)

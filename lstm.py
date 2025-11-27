import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, n_labels, hidden=256, num_layers=5,
                 bidirectional=False, pad_idx=0):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embed = nn.Embedding(vocab_size, 256, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3 if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden * (2 if bidirectional else 1), n_labels)

    def forward(self, x):
        x = self.embed(x)
        output, (h, c) = self.lstm(x)
        last_hidden = h[-1]   # (B, H)
        logits = self.fc(last_hidden)
        return logits

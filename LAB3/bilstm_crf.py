import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, num_labels, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 256, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            bidirectional=True,
            num_layers=5,
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Linear(256*2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, labels=None, mask=None):
        x = self.embed(input_ids)
        x, _ = self.lstm(x)
        emissions = self.fc(x)

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask, reduction="mean")
            return loss

        # decode
        return self.crf.decode(emissions, mask=mask)

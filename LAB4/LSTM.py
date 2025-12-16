import torch
import torch.nn as nn

class Seq2SeqLSTM(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, pad_idx):
        super().__init__()
        self.hidden_size = 256
        self.num_layers = 3
        self.src_embedding = nn.Embedding(src_vocab_size, 256, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, 256, padding_idx=pad_idx)
        self.encoder = nn.LSTM(256, 256, 3, batch_first=True)
        self.decoder = nn.LSTM(256, 256, 3, batch_first=True)
        self.fc = nn.Linear(256, tgt_vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, src, tgt):
        embedded_src = self.src_embedding(src)
        _, (h, c) = self.encoder(embedded_src)
        embedded_tgt = self.tgt_embedding(tgt[:, :-1])
        outputs, _ = self.decoder(embedded_tgt, (h, c))
        logits = self.fc(outputs)
        return logits

    def compute_loss(self, logits, tgt):
        return self.loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))

import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2SeqAttention2(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, pad_idx):
        super().__init__()
        self.hidden_size = 256
        self.num_layers = 3

        self.src_embedding = nn.Embedding(src_vocab_size, 256, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, 256, padding_idx=pad_idx)

        self.encoder = nn.LSTM(256, 256, num_layers=3, batch_first=True)

        self.decoder = nn.LSTM(256, 256, num_layers=3, batch_first=True)

        # Attention
        self.attn = nn.Linear(256 + 256, 256)
        self.v = nn.Linear(256, 1, bias=False)

        self.fc = nn.Linear(256, tgt_vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, src, tgt):
        # src: (B, S), tgt: (B, T)
        embedded_src = self.src_embedding(src)
        encoder_outputs, (h, c) = self.encoder(embedded_src)  # encoder_outputs: (B,S,H)

        embedded_tgt = self.tgt_embedding(tgt[:, :-1])
        batch_size, tgt_len, _ = embedded_tgt.size()
        outputs = []

        hidden, cell = h, c

        for t in range(tgt_len):
            y_t = embedded_tgt[:, t].unsqueeze(1)  # (B,1,H)

            # compute attention weights
            repeat_hidden = hidden[-1].unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
            energy = torch.tanh(self.attn(torch.cat((repeat_hidden, encoder_outputs), dim=2)))
            attn_weights = F.softmax(self.v(energy).squeeze(2), dim=1)  # (B,S)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (B,1,H)

            # LSTM input: y_t + context
            lstm_input = y_t + context
            output, (hidden, cell) = self.decoder(lstm_input, (hidden, cell))
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)  # (B,T,H)
        logits = self.fc(outputs)  # (B,T,V)
        return logits

    def compute_loss(self, logits, tgt):
        return self.loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))

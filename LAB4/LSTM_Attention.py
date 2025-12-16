import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2SeqAttention(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, pad_idx, hidden_size=256, num_layers=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding
        self.src_embedding = nn.Embedding(src_vocab_size, hidden_size, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, hidden_size, padding_idx=pad_idx)

        # Encoder & Decoder
        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        # Attention
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

        # Output
        self.fc = nn.Linear(hidden_size, tgt_vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, src, tgt):
        # src: (B, S), tgt: (B, T)
        embedded_src = self.src_embedding(src)
        enc_outputs, (h, c) = self.encoder(embedded_src)

        embedded_tgt = self.tgt_embedding(tgt[:, :-1])
        batch_size, tgt_len, _ = embedded_tgt.size()
        dec_outputs = []

        dec_h, dec_c = h, c
        for t in range(tgt_len):
            dec_input = embedded_tgt[:, t:t+1, :]  # (B,1,H)
            dec_output, (dec_h, dec_c) = self.decoder(dec_input, (dec_h, dec_c))  # (B,1,H)

            # Attention
            attn_weights = self._attention(enc_outputs, dec_output)  # (B, S)
            context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs)  # (B,1,H)
            dec_output = dec_output + context  # combine

            dec_outputs.append(dec_output)

        outputs = torch.cat(dec_outputs, dim=1)  # (B,T,H)
        logits = self.fc(outputs)  # (B,T,V)
        return logits

    def _attention(self, enc_outputs, dec_hidden):
        # enc_outputs: (B,S,H), dec_hidden: (B,1,H)
        seq_len = enc_outputs.size(1)
        dec_hidden_exp = dec_hidden.expand(-1, seq_len, -1)  # (B,S,H)
        energy = torch.tanh(self.attn(torch.cat((dec_hidden_exp, enc_outputs), dim=2)))  # (B,S,H)
        attn_scores = self.v(energy).squeeze(2)  # (B,S)
        return F.softmax(attn_scores, dim=1)

    def compute_loss(self, logits, tgt):
        return self.loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))

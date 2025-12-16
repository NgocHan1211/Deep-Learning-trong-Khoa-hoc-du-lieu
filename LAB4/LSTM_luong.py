import torch
from torch.utils.data import DataLoader
from Dataset import PhoMTDataset, collate_fn
from Vocab import Vocab
from tqdm import tqdm
import torch.nn as nn
from rouge_score import rouge_scorer

class Seq2SeqLuong(nn.Module):
    def __init__(self, vocab: Vocab, d_model=256, n_encoder=2, n_decoder=2, dropout=0.1):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.n_encoder = n_encoder
        self.n_decoder = n_decoder
        
        self.src_embedding = nn.Embedding(vocab.total_src_tokens(), d_model, padding_idx=vocab.pad_idx)
        self.tgt_embedding = nn.Embedding(vocab.total_tgt_tokens(), d_model, padding_idx=vocab.pad_idx)
        
        self.encoder = nn.LSTM(d_model, d_model, n_encoder, batch_first=True, bidirectional=True, dropout=dropout)
        self.decoder = nn.LSTM(d_model + 2*d_model, 2*d_model, n_decoder, batch_first=True)
        
        # Luong attention
        self.attn = nn.Linear(2*d_model, 2*d_model)
        
        self.fc_out = nn.Linear(2*d_model, vocab.total_tgt_tokens())
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    def forward(self, src, tgt):
        bs, src_len = src.size()
        bs, tgt_len = tgt.size()
        device = src.device

        # encoder
        enc_emb = self.src_embedding(src)
        enc_outputs, (h, c) = self.encoder(enc_emb) 
        
        # decoder initial state
        dec_h = torch.zeros(self.n_decoder, bs, 2*self.d_model).to(device)
        dec_c = torch.zeros(self.n_decoder, bs, 2*self.d_model).to(device)
        
        logits = []
        tgt_emb = self.tgt_embedding(tgt[:, :-1])
        
        for t in range(tgt_emb.size(1)):
            y_t = tgt_emb[:, t, :].unsqueeze(1)  
            
            # compute attention 
            dec_h_last = dec_h[-1].unsqueeze(1)  
            score = torch.bmm(dec_h_last, enc_outputs.transpose(1,2)) 
            attn_weights = torch.softmax(score, dim=-1) 
            context = torch.bmm(attn_weights, enc_outputs)  
            
            dec_input = torch.cat([y_t, context], dim=-1)
            _, (dec_h, dec_c) = self.decoder(dec_input, (dec_h, dec_c))
            
            logit = self.fc_out(dec_h[-1])
            logits.append(logit.unsqueeze(1))
        
        logits = torch.cat(logits, dim=1)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:,1:].reshape(-1))
        return loss, logits

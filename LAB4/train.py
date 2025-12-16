import torch
from torch.utils.data import DataLoader
from Dataset import PhoMTDataset, collate_fn
from Vocab import Vocab
from LSTM import Seq2SeqLSTM
from tqdm import tqdm
import pandas as pd
import json
from rouge_score import rouge_scorer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = '/content/drive/MyDrive/DL-TH4/data/'

# Vocab
src_vocab = Vocab("english", "vietnamese")
tgt_vocab = Vocab("english", "vietnamese")
src_vocab.make_vocab(data_path)
tgt_vocab.make_vocab(data_path)

# Dataset
train_dataset = PhoMTDataset(data_path + 'small-train.json', src_vocab, tgt_vocab)
dev_dataset   = PhoMTDataset(data_path + 'small-dev.json', src_vocab, tgt_vocab)
test_dataset  = PhoMTDataset(data_path + 'small-test.json', src_vocab, tgt_vocab)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
dev_loader   = DataLoader(dev_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# Model
model = Seq2SeqLSTM(src_vocab.total_src_tokens(), tgt_vocab.total_tgt_tokens(), src_vocab.pad_idx).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train
num_epochs = 5
for epoch in range(1, num_epochs+1):
    model.train()
    total_loss = 0
    for en_batch, vi_batch in tqdm(train_loader):
        en_batch, vi_batch = en_batch.to(device), vi_batch.to(device)
        optimizer.zero_grad()
        logits = model(en_batch, vi_batch)
        loss = model.compute_loss(logits, vi_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}/{num_epochs}, Avg Loss: {total_loss/len(train_loader):.4f}")

# Save model
model_path = '/content/drive/MyDrive/DL-TH4/seq2seq_20k_train.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Evaluate ROUGE-L
def evaluate(loader, dataset_name):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    with torch.no_grad():
        for en_batch, vi_batch in loader:
            en_batch, vi_batch = en_batch.to(device), vi_batch.to(device)
            logits = model(en_batch, vi_batch)
            pred_ids = torch.argmax(logits, dim=-1)
            for pred, ref in zip(pred_ids, vi_batch):
                pred_sent = tgt_vocab.decode_sentence(pred, "vietnamese")
                ref_sent  = tgt_vocab.decode_sentence(ref, "vietnamese")
                scores.append(scorer.score(ref_sent, pred_sent)['rougeL'].fmeasure)
    avg_score = sum(scores)/len(scores)
    print(f"ROUGE-L trung bình trên {dataset_name}: {avg_score:.4f}")
    return avg_score

evaluate(dev_loader, "Dev")
evaluate(test_loader, "Test")

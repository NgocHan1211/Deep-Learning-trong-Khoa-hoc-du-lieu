import torch
from torch.utils.data import DataLoader
from Dataset import PhoMTDataset, collate_fn
from Vocab import Vocab
from LSTM import Seq2SeqLSTM
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vocab
data_path = '/content/drive/MyDrive/DL-TH4/data/'
src_vocab = Vocab("english", "vietnamese")
tgt_vocab = Vocab("english", "vietnamese")
src_vocab.make_vocab(data_path)
tgt_vocab.make_vocab(data_path)

# Dataset nhỏ sample
train_dataset = PhoMTDataset(data_path + 'train.json', src_vocab, tgt_vocab)
dev_dataset   = PhoMTDataset(data_path + 'dev.json', src_vocab, tgt_vocab)
train_dataset.data = train_dataset.data[:2000]
dev_dataset.data   = dev_dataset.data[:500]

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=0)
dev_loader   = DataLoader(dev_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0)

# Model
model = Seq2SeqLSTM(src_vocab.total_src_tokens(), tgt_vocab.total_tgt_tokens(), src_vocab.pad_idx).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train 3 epoch
num_epochs = 3
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
torch.save(model.state_dict(), '/content/drive/MyDrive/DL-TH4/seq2seq_sample.pth')

# Eval ROUGE-L
from rouge_score import rouge_scorer
model.eval()
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
scores = []
for en_batch, vi_batch in dev_loader:
    en_batch, vi_batch = en_batch.to(device), vi_batch.to(device)
    with torch.no_grad():
        logits = model(en_batch, vi_batch)
        pred_ids = torch.argmax(logits, dim=-1)
    for pred, ref in zip(pred_ids, vi_batch):
        pred_sent = tgt_vocab.decode_sentence(pred, "vietnamese")
        ref_sent  = tgt_vocab.decode_sentence(ref, "vietnamese")
        scores.append(scorer.score(ref_sent, pred_sent)['rougeL'].fmeasure)
print("ROUGE-L trung bình:", sum(scores)/len(scores))

import torch
from torch.utils.data import DataLoader
from Dataset import PhoMTDataset, collate_fn
from Vocab import Vocab
from LSTM_luong import Seq2SeqLuong
from tqdm import tqdm
from rouge_score import rouge_scorer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = '/content/drive/MyDrive/DL-TH4/data/'

# vocab
src_vocab = Vocab("english","vietnamese")
tgt_vocab = Vocab("english","vietnamese")
src_vocab.make_vocab(data_path)
tgt_vocab.make_vocab(data_path)

# dataset
train_dataset = PhoMTDataset(data_path + 'small-train.json', src_vocab, tgt_vocab)
dev_dataset   = PhoMTDataset(data_path + 'small-dev.json', src_vocab, tgt_vocab)
test_dataset  = PhoMTDataset(data_path + 'small-test.json', src_vocab, tgt_vocab)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
dev_loader   = DataLoader(dev_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# model
model = Seq2SeqLuong(vocab=src_vocab).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# train
num_epochs = 5
for epoch in range(1, num_epochs+1):
    model.train()
    total_loss = 0
    for src_batch, tgt_batch in tqdm(train_loader):
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
        optimizer.zero_grad()
        loss, _ = model(src_batch, tgt_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}/{num_epochs}, Avg Loss: {total_loss/len(train_loader):.4f}")

# save
torch.save(model.state_dict(), '/content/drive/MyDrive/DL-TH4/seq2seq_luong.pth')

# evaluate ROUGE-L
def evaluate(loader, name):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    with torch.no_grad():
        for src_batch, tgt_batch in loader:
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
            _, logits = model(src_batch, tgt_batch)
            pred_ids = torch.argmax(logits, dim=-1)
            for pred, ref in zip(pred_ids, tgt_batch):
                pred_sent = tgt_vocab.decode_sentence(pred, "vietnamese")
                ref_sent  = tgt_vocab.decode_sentence(ref, "vietnamese")
                scores.append(scorer.score(ref_sent, pred_sent)['rougeL'].fmeasure)
    avg = sum(scores)/len(scores)
    print(f"ROUGE-L trung bình trên {name}: {avg:.4f}")
    return avg

evaluate(dev_loader, "Dev")
evaluate(test_loader, "Test")

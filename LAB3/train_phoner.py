import os
import json
import torch
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score, classification_report

from phoner_dataset import NERDataset, collate_fn
from bilstm_crf import BiLSTM_CRF

DATA = "/content/drive/MyDrive/DL-TH3/PhoNER_COVID19/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_jsonlines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

train_json = read_jsonlines(os.path.join(DATA, "train_word.json"))
dev_json   = read_jsonlines(os.path.join(DATA, "dev_word.json"))
test_json  = read_jsonlines(os.path.join(DATA, "test_word.json"))

words = set()
labels = set()
for dset in [train_json, dev_json, test_json]:
    for item in dset:
        words.update([w.lower() for w in item["words"]])
        labels.update(item["tags"])

word2idx = {"<pad>":0, "<unk>":1}
for w in words:
    word2idx[w] = len(word2idx)

label2idx = {label: i for i, label in enumerate(sorted(labels))}
idx2label = {i: l for l, i in label2idx.items()}

train_ds = NERDataset(os.path.join(DATA, "train_word.json"), word2idx, label2idx)
dev_ds   = NERDataset(os.path.join(DATA, "dev_word.json"), word2idx, label2idx)
test_ds  = NERDataset(os.path.join(DATA, "test_word.json"), word2idx, label2idx)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
dev_loader   = DataLoader(dev_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

model = BiLSTM_CRF(len(word2idx), len(label2idx), pad_idx=0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_f1 = 0
patience = 3
wait = 0
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)

        loss = model(ids, labels, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch} - Train loss: {total_loss:.4f}")

    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for batch in dev_loader:
            ids = batch["input_ids"].to(device)
            labels = batch["labels"]
            mask = batch["mask"].to(device)
            pred = model(ids, mask=mask)

            for p, g, m in zip(pred, labels, mask.cpu()):
                real_len = sum(m).item()
                preds.append([idx2label[x] for x in p[:real_len]])
                golds.append([idx2label[x.item()] for x in g[:real_len]])

    f1 = f1_score(golds, preds)
    print(f"Dev F1 = {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        wait = 0
        torch.save(model.state_dict(), "/content/drive/MyDrive/DL-TH3/best_model.pt")
        print("Model improved, saving checkpoint.")
    else:
        wait += 1
        print(f"No improvement. Wait {wait}/{patience}")

    if wait >= patience:
        print("Early stopping triggered!")
        break

print("\nTEST SET")
model.load_state_dict(torch.load("/content/drive/MyDrive/DL-TH3/best_model.pt"))
model.eval()

preds, golds = [], []
with torch.no_grad():
    for batch in test_loader:
        ids = batch["input_ids"].to(device)
        labels = batch["labels"]
        mask = batch["mask"].to(device)
        pred = model(ids, mask=mask)
        for p, g, m in zip(pred, labels, mask.cpu()):
            real_len = sum(m).item()
            preds.append([idx2label[x] for x in p[:real_len]])
            golds.append([idx2label[x.item()] for x in g[:real_len]])

print("Test F1:", f1_score(golds, preds))
print(classification_report(golds, preds))

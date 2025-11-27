import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from uit_vsfc import Vocab, VSFCDataset, collate_fn
from lstm import LSTMClassifier

DATA_FOLDER = "/content/drive/MyDrive/DL-TH3/UIT-VSFC"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
vocab = Vocab(DATA_FOLDER)
train_ds = VSFCDataset(os.path.join(DATA_FOLDER, "UIT-VSFC-train.json"), vocab)
dev_ds   = VSFCDataset(os.path.join(DATA_FOLDER, "UIT-VSFC-dev.json"), vocab)
test_ds  = VSFCDataset(os.path.join(DATA_FOLDER, "UIT-VSFC-test.json"), vocab)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
dev_loader   = DataLoader(dev_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Model
model = LSTMClassifier(
    vocab_size=len(vocab),
    n_labels=vocab.n_labels(),
    hidden=256,
    num_layers=5,
    bidirectional=False,
    pad_idx=0
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# early stopping
patience = 3
best_f1 = 0
bad_epochs = 0

for epoch in range(20):
    model.train()
    total_loss = 0

    for batch in train_loader:
        x = batch["input_ids"].to(device)
        y = batch["label_ids"].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()   # ✔ phải cộng!

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} loss = {avg_loss:.4f}")

    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for batch in dev_loader:
            x = batch["input_ids"].to(device)
            y = batch["label_ids"].to(device)

            logits = model(x)
            pred = torch.argmax(logits, dim=-1)
            preds.extend(pred.tolist())
            golds.extend(y.tolist())

    f1 = f1_score(golds, preds, average="macro")
    print(f"Dev F1: {f1:.4f}")

    # early stopping
    if f1 > best_f1:
        best_f1 = f1
        bad_epochs = 0
        torch.save(model.state_dict(), "/content/drive/MyDrive/DL-TH3/best_lstm.pt")
        print("Saved best model.")
    else:
        bad_epochs += 1
        print(f"No improvement: {bad_epochs}/{patience}")

        if bad_epochs >= patience:
            print("Early stopping triggered!")
            break

# test
model.load_state_dict(torch.load("/content/drive/MyDrive/DL-TH3/best_lstm.pt"))
model.eval()

preds, golds = [], []
with torch.no_grad():
    for batch in test_loader:
        x = batch["input_ids"].to(device)
        y = batch["label_ids"].to(device)

        logits = model(x)
        pred = torch.argmax(logits, dim=-1)
        preds.extend(pred.tolist())
        golds.extend(y.tolist())

print("Test F1:", f1_score(golds, preds, average="macro"))

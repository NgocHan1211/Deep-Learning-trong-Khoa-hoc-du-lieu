import json
import torch
from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, json_path, word2idx, label2idx):
        self.sentences = []
        self.labels = []

        # đọc JSON Lines
        with open(json_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

        for sample in data:
            words = sample["words"]
            tags = sample["tags"]

            self.sentences.append([word2idx.get(w.lower(), word2idx["<unk>"]) for w in words])
            self.labels.append([label2idx[t] for t in tags])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            "tokens": self.sentences[idx],
            "labels": self.labels[idx]
        }

def collate_fn(batch, pad_idx=0):
    max_len = max(len(x["tokens"]) for x in batch)

    input_ids = []
    tag_ids = []
    mask = []

    for item in batch:
        tokens = item["tokens"]
        labels = item["labels"]

        pad_len = max_len - len(tokens)

        input_ids.append(tokens + [pad_idx] * pad_len)
        tag_ids.append(labels + [0] * pad_len)  # 0 dùng cho padding
        mask.append([1]*len(tokens) + [0]*pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(tag_ids, dtype=torch.long),
        "mask": torch.tensor(mask, dtype=torch.bool)
    }

import torch
from torch.utils.data import Dataset
import json
import os
import string
from torch.nn import functional as F

def collate_fn(items: list, pad_idx=0) -> dict:
    input_ids = [item["input_ids"] for item in items]
    max_len = max([x.shape[0] for x in input_ids])

    input_ids = [
        F.pad(x, pad=(0, max_len - x.shape[0]), mode="constant", value=pad_idx).unsqueeze(0)
        for x in input_ids
    ]
    input_ids = torch.cat(input_ids, dim=0)

    label_ids = torch.tensor([item["label"] for item in items], dtype=torch.long)
    return {"input_ids": input_ids, "label_ids": label_ids}


class Vocab:
    def __init__(self, path: str):
        all_words = set()
        labels = set()
        for filename in os.listdir(path):
            if not filename.endswith(".json"):
                continue
            data = json.load(open(os.path.join(path, filename)))
            for item in data:
                sentence = self.preprocess_sentence(item["sentence"])
                all_words.update(sentence.split())
                labels.add(item["topic"])

        self.bos = "<s>"
        self.pad = "<p>"

        self.w2i = {word: idx for idx, word in enumerate(all_words, start=2)}
        self.w2i[self.pad] = 0
        self.w2i[self.bos] = 1
        self.i2w = {idx: word for word, idx in self.w2i.items()}

        self.l2i = {label: idx for idx, label in enumerate(labels)}
        self.i2l = {idx: label for label, idx in self.l2i.items()}

    def n_labels(self):
        return len(self.l2i)

    def __len__(self):
        return len(self.w2i)

    def preprocess_sentence(self, sentence: str) -> str:
        translator = str.maketrans("", "", string.punctuation)
        sentence = sentence.lower()
        sentence = sentence.translate(translator)
        return sentence


class VSFCDataset(Dataset):
    def __init__(self, json_path, vocab: Vocab):
        self.data = json.load(open(json_path))
        self.vocab = vocab

    def encode_sentence(self, sentence):
        sentence = self.vocab.preprocess_sentence(sentence)
        tokens = sentence.split()
        ids = [self.vocab.w2i.get(tok, 1) for tok in tokens]  # OOV=1
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = self.encode_sentence(item["sentence"])
        label = self.vocab.l2i[item["topic"]]
        return {"input_ids": input_ids, "label": label}

    def __len__(self):
        return len(self.data)

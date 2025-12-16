import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json

class PhoMTDataset(Dataset):
    def __init__(self, path, src_vocab, tgt_vocab):
        with open(path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        # Encode sẵn để DataLoader nhanh
        self.data = [{"en": src_vocab.encode_sentence(item["english"], "english"),
                      "vi": tgt_vocab.encode_sentence(item["vietnamese"], "vietnamese")}
                     for item in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    en = [item["en"] for item in batch]
    vi = [item["vi"] for item in batch]
    en = pad_sequence(en, batch_first=True, padding_value=0)
    vi = pad_sequence(vi, batch_first=True, padding_value=0)
    return en, vi

import os
import json
import torch
import re

class Vocab:
    def __init__(self, src_language: str, tgt_language: str):
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.initialize_special_tokens()

    def initialize_special_tokens(self):
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.specials = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

    def preprocess_sentence(self, sentence: str):
        sentence = sentence.lower()
        return re.findall(r"\w+", sentence)

    def make_vocab(self, path: str):
        src_words = set()
        tgt_words = set()
        for file in os.listdir(path):
            data = json.load(open(os.path.join(path, file), encoding="utf-8"))
            for item in data:
                src_words.update(self.preprocess_sentence(item[self.src_language]))
                tgt_words.update(self.preprocess_sentence(item[self.tgt_language]))
        src_itos = self.specials + list(src_words)
        tgt_itos = self.specials + list(tgt_words)
        self.src_itos = {i: tok for i, tok in enumerate(src_itos)}
        self.src_stoi = {tok: i for i, tok in enumerate(src_itos)}
        self.tgt_itos = {i: tok for i, tok in enumerate(tgt_itos)}
        self.tgt_stoi = {tok: i for i, tok in enumerate(tgt_itos)}

    def encode_sentence(self, sentence: str, language: str):
        tokens = self.preprocess_sentence(sentence)
        stoi = self.src_stoi if language == self.src_language else self.tgt_stoi
        vec = [self.bos_idx] + [stoi.get(tok, self.unk_idx) for tok in tokens] + [self.eos_idx]
        return torch.tensor(vec, dtype=torch.long)

    def decode_sentence(self, vec: torch.Tensor, language: str):
        ids = vec.tolist()
        itos = self.src_itos if language == self.src_language else self.tgt_itos
        words = []
        for idx in ids:
            if idx == self.eos_idx:
                break
            words.append(itos[idx])
        return " ".join(words)

    def total_src_tokens(self):
        return len(self.src_itos)

    def total_tgt_tokens(self):
        return len(self.tgt_itos)

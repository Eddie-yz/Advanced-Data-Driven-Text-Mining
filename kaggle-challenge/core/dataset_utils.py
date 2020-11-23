import numpy as np
import torch
from torch.utils.data import Dataset


class TextDatasetBert(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


class TextDatasetLSTM(Dataset):
    def __init__(self, docs, labels, word2ind, max_length=400):
        self.docs = docs
        self.labels = labels
        self.word2ind = word2ind
        self.max_length = max_length
    
    def encode_sentence(self, text):
        encoded = np.tile(self.word2ind['PAD'], self.max_length)
        enc1 = np.array([self.word2ind.get(word, self.word2ind['UNK']) for word in text])
        length = min(self.max_length, len(enc1))
        encoded[:length] = enc1[:length]
        return encoded

    def __getitem__(self, idx):
        return self.encode_sentence(self.docs[idx]), self.labels[idx]

    def __len__(self):
        return len(self.labels)

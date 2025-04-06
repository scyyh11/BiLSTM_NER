from torch.utils.data import Dataset
import torch


class conll2003_dataset(Dataset):
    def __init__(self, hf_dataset, word_to_idx):
        self.sentences = []
        self.labels = []
        for example in hf_dataset:
            self.sentences.append([word_to_idx.get(token, word_to_idx['<UNK>']) for token in example['tokens']])
            self.labels.append(example['ner_tags'])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx]), torch.tensor(self.labels[idx])

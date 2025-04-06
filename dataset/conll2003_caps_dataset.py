import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils.handle_caps import get_case_id


class conll2003_caps_dataset(Dataset):
    def __init__(self, file_path, word_to_idx, label_to_idx):
        self.sentences, self.labels, self.caps_cases = [], [], []

        with open(file_path, 'r', encoding='utf-8') as file:
            sentence, labels, caps_case = [], [], []
            for line in file:
                if line.strip():
                    _, word, label = line.strip().split()
                    sentence.append(word_to_idx.get(word, word_to_idx['<UNK>']))
                    labels.append(label_to_idx[label])
                    caps_case.append(get_case_id(word))
                else:
                    if sentence:
                        self.sentences.append(sentence)
                        self.labels.append(labels)
                        self.caps_cases.append(caps_case)
                        sentence, labels, caps_case = [], [], []

            if sentence:
                self.sentences.append(sentence)
                self.labels.append(labels)
                self.caps_cases.append(caps_case)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx]), torch.tensor(self.labels[idx]), torch.tensor(self.caps_cases[idx])

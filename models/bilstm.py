import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        super(BiLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        linear_out = self.fc(lstm_out)
        output = self.classifier(self.elu(linear_out))
        return output
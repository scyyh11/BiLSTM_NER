import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, label_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(output_dim, label_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        linear_out = self.fc(lstm_out)
        activated = self.elu(linear_out)
        output = self.classifier(activated)
        return output

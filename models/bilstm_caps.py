import torch
import torch.nn as nn


class BiLSTM_Caps(nn.Module):
    def __init__(self, label_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout,
                 embeddings=None, caps_embeddings=5):
        super(BiLSTM_Caps, self).__init__()

        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float)
        self.embedding = nn.Embedding.from_pretrained(embeddings_tensor, freeze=False)
        self.case_embedding = nn.Embedding(5, caps_embeddings)
        lstm_input_dim = embedding_dim + caps_embeddings

        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(output_dim, label_size)

    def forward(self, word, caps):
        x = torch.cat((self.embedding(word), self.case_embedding(caps)), dim=2)
        lstm_out, _ = self.lstm(x)
        linear_out = self.fc(lstm_out)
        activated = self.elu(linear_out)
        output = self.classifier(activated)
        return output






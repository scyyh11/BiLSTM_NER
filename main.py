from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset

from models.bilstm import BiLSTM
from dataset.conll2003_dataset import conll2003_dataset
from utils.vocab import build_vocab_from_dataset
from utils.collate_fn import collate_fn
from training.train import train_model
from training.evaluate import evaluate_model
import config as cfg


def main():
    dataset = load_dataset("conll2003", trust_remote_code=True)
    train_data = dataset['train']
    val_data = dataset['validation']

    word_to_idx, label_to_idx = build_vocab_from_dataset(train_data)

    train_dataset = conll2003_dataset(train_data, word_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    val_dataset = conll2003_dataset(val_data, word_to_idx)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, collate_fn=collate_fn)

    model = BiLSTM(vocab_size=len(word_to_idx),
                   label_size=len(label_to_idx),
                   embedding_dim=cfg.EMBEDDING_DIM,
                   hidden_dim=cfg.HIDDEN_DIM,
                   output_dim=cfg.OUTPUT_DIM,
                   num_layers=cfg.NUM_LAYERS,
                   dropout=cfg.DROPOUT).to(cfg.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.LR, momentum=cfg.MOMENTUM)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.STEP_SIZE, gamma=cfg.GAMMA)

    train_model(model, train_loader, criterion, optimizer, scheduler, cfg.EPOCHS, cfg.DEVICE, len(label_to_idx))
    evaluate_model(model, val_loader, cfg.DEVICE)

if __name__ == '__main__':
    main()

import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
EPOCHS = 50

NUM_LAYERS = 2
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 128
DROPOUT = 0.33

LR = 0.1
MOMENTUM = 0.9
STEP_SIZE = 20
GAMMA = 0.1

import torch


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
NUM_LAYERS = 1
DROP_RATE = 0.33

EPOCHS = 50
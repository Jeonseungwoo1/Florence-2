import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = 'microsoft/Florence-2-large'
BATCH_SIZE = 6
NUM_EPOCHS = 7
LEARNING_RATE= 1e-6
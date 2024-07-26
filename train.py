import torch
from transformers import AdamW
from .config import NUM_EPOCHS, LEARNING_RATE

def train_model(model, train_loader):
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(NUM_EPOCHS):
        model.train()
        for inputs, _ in train_loader:
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return model

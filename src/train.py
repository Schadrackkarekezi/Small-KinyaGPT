
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tokenizers import Tokenizer
from src.model import GPT, GPTConfig


def create_blocks(data, block_size):
    n_blocks = len(data) // block_size
    data = data[:n_blocks * block_size]
    return data.reshape(n_blocks, block_size)

def make_dataset(data):
    x = data[:, :-1]
    y = data[:, 1:]
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))

def train_model(tokenizer_path="data/kinyarwanda_bpe.json", block_size=512, num_epochs=8):
    """Trains the KinyaGPT model on preprocessed tokenized data"""
    tokenizer = Tokenizer.from_file(tokenizer_path)

    all_ids = np.load("data/all_ids.npy")

    split_idx = int(0.9 * len(all_ids))
    train_ids, val_ids = all_ids[:split_idx], all_ids[split_idx:]

    train_blocks = create_blocks(train_ids, block_size)
    val_blocks = create_blocks(val_ids, block_size)

    train_loader = DataLoader(make_dataset(train_blocks), batch_size=32, shuffle=True)
    val_loader   = DataLoader(make_dataset(val_blocks), batch_size=32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(GPTConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).long()
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).long()
                _, loss = model(x, y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}  Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}")


if __name__ == "__main__":
    train_model()

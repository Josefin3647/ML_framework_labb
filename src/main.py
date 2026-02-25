import torch
from src.dataset import get_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

train_loader, test_loader = get_dataloaders(
    data_root="data",
    batch_size=128,
    download=False
)
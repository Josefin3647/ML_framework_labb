from pathlib import Path
from src.config import DATA_PATH, NUM_WORKERS
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloaders(
    data_root=DATA_PATH,
    batch_size=64,
    num_workers=NUM_WORKERS
):
    """
    Load CIFAR-10 using torchvision and return train/test DataLoaders.

    Args:
        data_root (str or Path): Path to data folder (default: "data")
        batch_size (int): Batch size
        num_workers (int): Number of CPU workers for DataLoader
        download (bool): Whether to download dataset if not present
    
    Returns:
    Train and test loaders

    """

    data_root = Path(data_root)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader

def get_eda_dataset(
    train=True,
    data_root=DATA_PATH
):
    """
    Load CIFAR-10 without augmentation or normalization.
    Useful for EDA and visualization.
    """

    data_root = Path(data_root)

    eda_transform = transforms.ToTensor()

    dataset = datasets.CIFAR10(
        root=data_root,
        train=train,
        download=True,
        transform=eda_transform
    )

    return dataset
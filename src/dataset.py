from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(
    data_root="data",
    batch_size=64,
    num_workers=2,
    download=False
):
    """
    Load CIFAR-10 using torchvision and return train/test DataLoaders.

    Args:
        data_root (str or Path): Path to data folder (default: "data")
        batch_size (int): Batch size
        num_workers (int): Number of CPU workers for DataLoader
        download (bool): Whether to download dataset if not present
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
        download=download,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=download,
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.constants import DATA_PATH


def get_dataloaders(data_root="data", batch_size=64):
    """
    Load CIFAR-10 and return train/test DataLoaders.
    """

    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(
        root=DATA_PATH,
        train=True,
        download=False,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_PATH,
        train=False,
        download=False,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader

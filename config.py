from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data"

NUM_WORKERS = 2
NUM_CLASSES = 10

EXPERIMENTS = [
    {
        "name": "baseline",
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 64,
    },
    {
        "name": "low_lr",
        "epochs": 15,
        "lr": 0.0003,
        "batch_size": 64,
    },
    {
        "name": "more_epochs",
        "epochs": 20,
        "lr": 0.001,
        "batch_size": 128,
    },
]
import torch
import config
from dataset import get_dataloaders
from model import SimpleCNN
from train import train

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    results = []

    for exp in config.EXPERIMENTS:

        print("\n==============================")
        print(f"Running experiment: {exp['name']}")
        print("==============================")

        train_loader, test_loader = get_dataloaders(
            data_root=config.DATA_ROOT,
            batch_size=exp["batch_size"],
            num_workers=config.NUM_WORKERS,
            download=config.DOWNLOAD,
        )

        model = SimpleCNN(num_classes=config.NUM_CLASSES)
        model.to(device)

        final_acc = train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=exp["epochs"],
            lr=exp["lr"],
        )

        results.append(
            {
                "name": exp["name"],
                "acc": final_acc,
            }
        )

    print("\n===== RESULTS =====")

    for r in results:
        print(f"{r['name']}: {r['acc']:.3f}")

    return results

if __name__ == "__main__":
    main()
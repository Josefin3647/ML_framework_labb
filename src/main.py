import json
import os

import torch
from src import config
from src.dataset import get_dataloaders
from src.model import SimpleCNN
from src.train import train


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    os.makedirs("logs", exist_ok=True)

    results = []
    all_results = {}

    for exp in config.EXPERIMENTS:
        print("\n==============================")
        print(f"Running experiment: {exp['name']}")
        print("==============================")

        train_loader, test_loader = get_dataloaders(
            data_root=config.DATA_PATH,
            batch_size=exp["batch_size"],
            num_workers=config.NUM_WORKERS,
        )

        model = SimpleCNN(num_classes=config.NUM_CLASSES).to(device)

        result = train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=exp["epochs"],
            lr=exp["lr"],
        )

        all_results[exp["name"]] = {
            "config": exp,
            "history": result,
        }

        with open(f"logs/{exp['name']}.json", "w") as f:
            json.dump(
                {
                    "config": exp,
                    "history": result,
                },
                f,
                indent=2,
            )

        results.append({
            "name": exp["name"],
            "train_loss": result["final_train_loss"],
            "test_loss": result["final_test_loss"],
            "acc": result["final_acc"],
        })

    with open("logs/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n===== RESULTS =====")
    print(f"{'Experiment':<15}{'Train Loss':<15}{'Test Loss':<15}{'Accuracy':<15}")
    print("-" * 60)

    for r in results:
        print(f"{r['name']:<15}{r['train_loss']:<15.4f}{r['test_loss']:<15.4f}{r['acc']:<15.4f}")

    return results


if __name__ == "__main__":
    main()
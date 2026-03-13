# train.py
import torch
import torch.nn as nn


@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluate accuracy on a dataloader.
    Returns accuracy in [0, 1].
    """
    model.eval()

    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / max(total, 1)


def train(
    model,
    train_loader,
    test_loader,
    device,
    epochs=10,
    lr=1e-3,
):
    """
    Trains model on train_loader and evaluates on test_loader each epoch.
    Returns final test accuracy (float in [0, 1]).

    This matches how main.py calls train(...).  :contentReference[oaicite:4]{index=4}
    """
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    final_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        test_acc = evaluate(model, test_loader, device)
        final_acc = test_acc
        best_acc = max(best_acc, test_acc)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.3f} | "
            f"test_acc={test_acc:.3f}"
        )

    print(f"Best test_acc: {best_acc:.3f}")
    return final_acc
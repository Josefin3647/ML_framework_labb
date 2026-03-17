import torch
import torch.nn as nn
import torch.optim as optim


def train(model, train_loader, test_loader, device, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []
    accuracies = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                test_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_test_loss = test_loss / len(test_loader)
        acc = correct / total

        test_losses.append(avg_test_loss)
        accuracies.append(acc)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"train loss: {avg_train_loss:.4f}, "
            f"test loss: {avg_test_loss:.4f}, "
            f"accuracy: {acc:.4f}"
        )

    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "accuracies": accuracies,
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "final_acc": accuracies[-1],
    }
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import json

from LeNet import LeNet
from mnist_dataset import MnistDataset, collate_fn

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(dataloader: DataLoader, model: nn.Module) -> dict:
    model.eval()
    predictions, trues = [], []

    with torch.no_grad():
        for item in dataloader:
            image: torch.Tensor = item["image"].to(device)
            label: torch.Tensor = item["label"].to(device)

            output: torch.Tensor = model(image)
            output = torch.argmax(output, dim=1)

            predictions.extend(output.cpu().numpy())
            trues.extend(label.cpu().numpy())

    return {
        "precision": precision_score(trues, predictions, average="macro", zero_division=0),
        "accuracy": accuracy_score(trues, predictions),
        "f1": f1_score(trues, predictions, average="macro", zero_division=0)
    }

# Huan luyen
if __name__ == "__main__":
    train_dataset = MnistDataset(
        image_path="/content/drive/MyDrive/DL-TH2/mnist-dataset/train-images-idx3-ubyte",
        label_path="/content/drive/MyDrive/DL-TH2/mnist-dataset/train-labels-idx1-ubyte"
    )

    test_dataset = MnistDataset(
        image_path="/content/drive/MyDrive/DL-TH2/mnist-dataset/t10k-images-idx3-ubyte",
        label_path="/content/drive/MyDrive/DL-TH2/mnist-dataset/t10k-labels-idx1-ubyte"
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = LeNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    EPOCH = 5
    best_score = 0
    best_metric = "f1"

    train_losses = []
    test_accuracies = []

    for epoch in range(EPOCH):
        model.train()
        epoch_loss = 0.0

        for item in train_dataloader:
            image: torch.Tensor = item["image"].to(device)
            label: torch.Tensor = item["label"].to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        metrics = evaluate(test_dataloader, model)
        avg_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_loss)
        test_accuracies.append(metrics["accuracy"])

        print(f"Epoch [{epoch+1}/{EPOCH}] "
              f"Loss: {avg_loss:.4f} "
              f"Acc: {metrics['accuracy']:.4f} "
              f"F1: {metrics['f1']:.4f}")

        if metrics[best_metric] > best_score:
            best_score = metrics[best_metric]
            torch.save(model.state_dict(), "/content/drive/MyDrive/DL-TH2/best_model.pth")
            print(f"Lưu mô hình tốt nhất tại epoch {epoch+1}")

    print("Đang lưu ")
    history_data = {
        "train_losses": train_losses,
        "test_accuracies": test_accuracies
    }

    with open("/content/drive/MyDrive/DL-TH2/training_history.json", "w") as f:
        json.dump(history_data, f)

    print("Đã lưu lịch sử vào 'training_history.json'")

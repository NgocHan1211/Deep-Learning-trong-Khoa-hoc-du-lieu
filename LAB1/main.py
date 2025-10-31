import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from mnist_dataset import collate_fn, MnistDataset
from perceptron_1_layer import Perceptron_1_layer
from perceptron_3_layer import Perceptron_3_layer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    losses = []
    for batch in dataloader:
        X = torch.tensor(batch["image"], dtype=torch.float32).to(device)
        y = torch.tensor(batch["label"], dtype=torch.long).to(device)

        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return losses


def evaluate(dataloader, model):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            X = torch.tensor(batch["image"], dtype=torch.float32).to(device)
            y = torch.tensor(batch["label"], dtype=torch.long).to(device)

            output = model(X)
            preds = torch.argmax(output, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='macro') * 100
    recall = recall_score(all_labels, all_preds, average='macro') * 100
    f1 = f1_score(all_labels, all_preds, average='macro') * 100

    print("\nThống kê kết quả:")
    print(classification_report(all_labels, all_preds, digits=4))

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


if __name__ == "__main__":
    model_type = input("Chọn mô hình (1layer / 3layer): ").strip().lower()

    # load data
    train_dataset = MnistDataset(
        image_path="/content/drive/MyDrive/DL-Thực hành/mnist-dataset/train-images-idx3-ubyte",
        label_path="/content/drive/MyDrive/DL-Thực hành/mnist-dataset/train-labels-idx1-ubyte"
    )
    test_dataset = MnistDataset(
        image_path="/content/drive/MyDrive/DL-Thực hành/mnist-dataset/t10k-images-idx3-ubyte",
        label_path="/content/drive/MyDrive/DL-Thực hành/mnist-dataset/t10k-labels-idx1-ubyte"
    )

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # khởi tạo
    if model_type == "1layer":
        model = Perceptron_1_layer(image_size=(28, 28), num_labels=10).to(device)
        checkpoint_path = "/content/drive/MyDrive/DL-Thực hành/checkpoint_perceptron_1_layer.pth"
    elif model_type == "3layer":
        model = Perceptron_3_layer(input_dim=28*28, num_classes=10).to(device)
        checkpoint_path = "/content/drive/MyDrive/DL-Thực hành/checkpoint_perceptron_3_layer.pth"
    else:
        raise ValueError("Chỉ được chọn '1layer' hoặc '3layer'.")

    print(f"\nĐang huấn luyện mô hình: {model_type}\n")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    EPOCHS = 5
    best_score = 0
    best_score_name = "f1"
    history_loss = []

    # train loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        losses = train(train_dataloader, model, loss_fn, optimizer)
        avg_loss = np.mean(losses)
        history_loss.append(avg_loss)
        print(f"Loss trung bình: {avg_loss:.4f}")

        scores = evaluate(test_dataloader, model)
        for score_name in scores:
            print(f"\t- {score_name}: {scores[score_name]:.2f}")

        current_score = scores[best_score_name]
        if current_score > best_score:
            best_score = current_score
            torch.save(model.state_dict(), checkpoint_path)
            print("Lưu mô hình tốt nhất.")

    print(f"\nHuấn luyện xong mô hình {model_type}. Best F1: {best_score:.2f}%")

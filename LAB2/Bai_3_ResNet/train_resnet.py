import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
import matplotlib.pyplot as plt
import json
import os

from resnet_model import create_resnet18
from image_dataset import ImageFolderDataset, collate_fn, get_train_transform, get_test_transform

device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("/kaggle/working/output_resnet", exist_ok=True)

def evaluate(dataloader: DataLoader, model: nn.Module) -> dict:
    model.eval()
    predictions, trues = [], []

    with torch.no_grad():
        for item in dataloader:
            image: torch.Tensor = item["image"].to(device)
            label: torch.Tensor = item["label"].to(device)

            output: torch.tensor = model(image)

            output = torch.argmax(output, dim=1)
            predictions.extend(output.cpu().numpy())
            trues.extend(label.cpu().numpy())

    return {
        "precision": precision_score(trues, predictions, average="macro", zero_division=0),
        "recall": recall_score(trues, predictions, average="macro", zero_division=0),
        "accuracy": accuracy_score(trues, predictions),
        "f1": f1_score(trues, predictions, average="macro", zero_division=0)
    }

# Huan luyen
if __name__ == "__main__":
    TRAIN_DIR = "/kaggle/input/vinafood21/VinaFood21/train" 
    TEST_DIR = "/kaggle/input/vinafood21/VinaFood21/test"
    NUM_CLASSES = 21 
    EPOCH = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    train_transform = get_train_transform()
    test_transform = get_test_transform()

    train_dataset = ImageFolderDataset(
        root_dir=TRAIN_DIR,
        transform=train_transform
    )

    test_dataset = ImageFolderDataset(
        root_dir=TEST_DIR,
        transform=test_transform
    )
    
    print(f"Tìm thấy {len(train_dataset)} ảnh train thuộc {NUM_CLASSES} lớp.")
    print(f"Tìm thấy {len(test_dataset)} ảnh test.")

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )

    model = create_resnet18(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    loss_fn = nn.CrossEntropyLoss()

    best_score = 0
    best_metric = "f1"
    train_losses = []
    test_metrics = [] 

    print("BẮT ĐẦU HUẤN LUYỆN ResNet-18")

    for epoch in range(EPOCH):
        model.train()
        epoch_loss = 0.0

        for item in train_dataloader:
            image: torch.Tensor = item["image"].to(device)
            label: torch.Tensor = item["label"].to(device)

            optimizer.zero_grad()
            
            # Vòng lặp đơn giản hơn GoogLeNet
            output = model(image)
            loss = loss_fn(output, label)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_dataloader)
        metrics = evaluate(test_dataloader, model)
        
        train_losses.append(avg_loss)
        test_metrics.append(metrics)

        print(f"Epoch [{epoch+1}/{EPOCH}] "
              f"Loss: {avg_loss:.4f} | "
              f"Acc: {metrics['accuracy']:.4f} | "
              f"Precision: {metrics['precision']:.4f} | "
              f"Recall: {metrics['recall']:.4f} | "
              f"F1: {metrics['f1']:.4f}")

        if metrics[best_metric] > best_score:
            best_score = metrics[best_metric]
            torch.save(model.state_dict(), "/kaggle/working/output_resnet/best_resnet18_model.pth")
            print(f"Lưu mô hình tốt nhất tại epoch {epoch+1}")
            
    print("HUẤN LUYỆN HOÀN TẤT")
    
    history_data = {
        "train_losses": train_losses,
        "test_metrics": test_metrics
    }
    
    with open("/kaggle/working/output_resnet/resnet_history.json", "w") as f:
        json.dump(history_data, f)
        
    print("Đã lưu lịch sử vào '/kaggle/working/output_resnet/resnet_history.json'")

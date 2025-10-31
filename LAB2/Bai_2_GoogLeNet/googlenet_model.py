import torch
import torch.nn as nn
import torchvision.models as models

def create_googlenet(num_classes: int = 10):
    """
    Tạo mô hình GoogLeNet và áp dụng các ràng buộc của đề bài.
    """
    
    model = models.googlenet(weights=None, num_classes=num_classes)
    model.conv1.conv = nn.Conv2d(
        3, 
        64, 
        kernel_size=(7, 7), 
        stride=(2, 2), 
        padding=(3, 3),  
        bias=False
    )

    for module in model.modules():
        if isinstance(module, nn.MaxPool2d):
            module.ceil_mode = True

    return model

if __name__ == "__main__":
    model = create_googlenet(num_classes=10)
    print("Cấu trúc mô hình GoogLeNet đã sửa đổi")
    print(f"Lớp Conv1: {model.conv1.conv}")
    print(f"Lớp MaxPool1: {model.maxpool1}")

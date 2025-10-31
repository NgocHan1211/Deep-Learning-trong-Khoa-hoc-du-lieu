import torch
import torch.nn as nn
import torchvision.models as models

def create_resnet18(num_classes: int = 21):
    """
    Tạo mô hình ResNet-18 và áp dụng Max Pooling (kernel = 3, stride = 2, paddding = 0).
    """
    model = models.resnet18(weights=None, num_classes=num_classes)

    model.maxpool = nn.MaxPool2d(
        kernel_size=3, 
        stride=2, 
        padding=0  
    )

    return model

if __name__ == "__main__":
    model = create_resnet18(num_classes=21)
    print("Cấu trúc mô hình ResNet-18 đã sửa đổi")
    
    print("\nconv1")
    print(model.conv1)
    
    print("\nMaxPool")
    print(model.maxpool)

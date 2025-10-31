import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from glob import glob

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_paths = sorted(glob(os.path.join(root_dir, "*", "*.jpg")))
        self.image_paths.extend(sorted(glob(os.path.join(root_dir, "*", "*.png"))))
        self.image_paths.extend(sorted(glob(os.path.join(root_dir, "*", "*.jpeg"))))
        
        self.class_names = sorted([d for d in os.listdir(root_dir) 
                                   if os.path.isdir(os.path.join(root_dir, d))])
        
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        self.labels = []
        for img_path in self.image_paths:
            class_name = os.path.basename(os.path.dirname(img_path))
            self.labels.append(self.class_to_idx[class_name])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"image": image, "label": torch.tensor(label, dtype=torch.long)}

# Hàm collate (gom batch)
def collate_fn(items: list[dict]) -> dict:
    images = [item["image"] for item in items]
    labels = [item["label"] for item in items]
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    return {"image": images, "label": labels}

normalize_transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
)

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_transform
    ])

def get_test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize_transform
    ])

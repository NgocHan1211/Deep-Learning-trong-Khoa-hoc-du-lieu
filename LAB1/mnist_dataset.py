import torch
from torch.utils.data import Dataset
import idx2numpy
import numpy as np

def collate_fn(items: list[dict]) -> dict:
    items = [{
        "image": np.expand_dims(item["image"], axis=0),
        "label": np.array(item["label"])
    } for item in items]

    items = {
        "image": np.stack([item["image"] for item in items], axis=0),
        "label": np.stack([item["label"] for item in items], axis=0)
    }
    return items


class MnistDataset(Dataset):
    def __init__(self, image_path: str, label_path: str):
        images = idx2numpy.convert_from_file(image_path)
        labels = idx2numpy.convert_from_file(label_path)

        self._data = [{
            "image": image.astype(np.float32) / 255.0,
            "label": int(label)
        } for image, label in zip(images, labels)]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        return self._data[idx]

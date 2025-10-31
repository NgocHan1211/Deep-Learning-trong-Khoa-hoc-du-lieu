import torch
from torch.utils.data import Dataset
import idx2numpy
import numpy as np


def collate_fn(items: list[dict]) -> dict:
    """
    Gom nhiều item (dict) thành batch tensor.
    Input: list các dict {"image": np.array, "label": int}
    Output: dict {"image": Tensor, "label": Tensor}
    """
    images = [np.expand_dims(item["image"], axis=0) for item in items]
    labels = [item["label"] for item in items]

    images = np.stack(images, axis=0)
    labels = np.array(labels, dtype=np.int64) 

    images = torch.tensor(images, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    return {"image": images, "label": labels}


class MnistDataset(Dataset):
    """
    Dataset đọc trực tiếp file MNIST dạng .idx
    Ví dụ:
      image_path = '/content/train-images.idx3-ubyte'
      label_path = '/content/train-labels.idx1-ubyte'
    """
    def __init__(self, image_path: str, label_path: str):
        images = idx2numpy.convert_from_file(image_path)
        labels = idx2numpy.convert_from_file(label_path)

        assert len(images) == len(labels), "Số lượng ảnh và nhãn không khớp!"
        self._data = [{
            "image": image.astype(np.float32) / 255.0,
            "label": int(label)
        } for image, label in zip(images, labels)]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        return self._data[idx]

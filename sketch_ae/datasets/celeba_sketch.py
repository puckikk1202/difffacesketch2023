import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from PIL import Image
import os

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor / 255)*2 - 1
    return processed_tensor

class CelebaSketch(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file = os.listdir(self.data_dir)[idx]
        img = Image.open(os.path.join(self.data_dir, file)).convert('L')
        if self.transform:
            img = self.transform(img)

        return img

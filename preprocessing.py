from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
import cv2
import os
import config
import numpy as np

"""
Training data augmentation
"""
train_tf = T.Compose([
    #  # Converts a torch.*Tensor of shape C x H x W or
    #  a numpy ndarray of shape H x W x C to a PIL Image while preserving the value range.
    T.ToPILImage(),
    T.Resize(128),
    # Crop the center part
    T.CenterCrop(128),
    # Data augment on colors
    T.ColorJitter(brightness=(0.95, 1.05),
                  contrast=(0.95,1.05),
                  saturation=(0.95, 1.05),
                  hue=0.05),
    # Random movement
    T.RandomAffine(5, translate=(0.01,0.1)),
    # Scale the image between [0,1] values
    T.ToTensor(),
    # Normalization
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5])
])

"""
Preprocessing of validation images
"""

val_tf = T.Compose([
    T.ToPILImage(),
    T.Resize(128),
    T.CenterCrop(128),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5])
])


class FruitImages(Dataset):

    """
    Class to fetch and preprocess the dataset
    """

    def __init__(self, files, transform=None):

        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, ix):

        fpath = self.files[ix]

        # Extract the class name
        path = Path(fpath)
        class_name = os.path.basename(path.parent.absolute())

        # Extract the image
        img = cv2.imread(path)
        img = torch.tensor(img).permute(2,0,1)

        return img.to(device), class_name

    def collate_fn(self, batch):

        imgs, classes = list(zip(*batch))

        self.id2int = dict.fromkeys(classes)

        value = 0
        for k, v in self.id2int.items():
            self.id2int[k] = value
            value += 1

        if self.transform:
            imgs = [self.transform(img)[None] for img in imgs]

        classes = [torch.tensor(self.id2int[key]) for key in classes]

        # Concatenates the given sequence of seq tensors in the given dimension.
        imgs, classes = [torch.cat(i).to(device) for i in [imgs, classes]]

        return imgs, classes






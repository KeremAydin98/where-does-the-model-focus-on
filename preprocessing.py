from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
import cv2
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
    # Random rotation
    T.RandomRotation(30),
    T.RandomHorizontalFlip(),
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

    def __init__(self, files, transform=None, device='cpu'):

        self.files = files
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, ix):

        fpath = self.files[ix]

        # Extract the class name
        path = Path(fpath)
        class_name = path.parent.parts[-1]

        # Extract the image
        img = cv2.imread(str(path))

        img = torch.tensor(img).permute(2,0,1)

        return img.to(self.device), class_name

    def collate_fn(self, batch):

        _imgs, classes = list(zip(*batch))

        id2int = dict.fromkeys(classes)

        value = 0
        for k, v in id2int.items():
            id2int[k] = value
            value += 1

        if self.transform:
            imgs = [torch.tensor(self.transform(img)) for img in _imgs]

        classes = [torch.tensor(id2int[key]) for key in classes]

        # Concatenates the given sequence of seq tensors in the given dimension.
        imgs, classes = [torch.stack(i).to(self.device) for i in [imgs, classes]]

        return imgs, classes, _imgs






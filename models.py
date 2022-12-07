import torch.nn as nn
import torch
import numpy as np

def convBlock(ni, no):

    """
    Created a batch of layers that are repeated

    :param ni: Input size
    :param no: output size
    :return: batch of layers
    """

    return nn.Sequential(nn.Conv2d(ni, no, kernel_size=3, padding=1),
                         nn.ReLU(),
                         nn.BatchNorm2d(no),
                         nn.MaxPool2d(2),
                         nn.Dropout(0.2))


class FruitClassifier(nn.Module):

    def __init__(self, id2int):

        super().__init__()

        self.model = nn.Sequential(convBlock(3,64),
                                   convBlock(64,128),
                                   convBlock(128,256),
                                   convBlock(256,512),
                                   convBlock(512, 64),
                                   nn.Flatten(),
                                   nn.Linear(1024,256),
                                   nn.Dropout(0.2),
                                   # inplace=True means that it will modify the input directly,
                                   # without allocating any additional output.
                                   nn.ReLU(inplace=True),
                                   # There is no need of softmax layer since CrossEntropyLoss does it anyway
                                   nn.Linear(256, id2int))

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):

        return self.model(x)

    def compute_metrics(self, preds, targets):

        # Calculating loss
        loss = self.loss_fn(preds, targets)

        # Calculating accuracy
        acc = (torch.max(preds, 1)[1] == targets).float().mean()

        return loss, acc


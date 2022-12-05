from torchvision import transforms as T

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

val_tfms = T.Compose([
    T.ToPILImage(),
    T.Resize(128),
    T.CenterCrop(128),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5])
])


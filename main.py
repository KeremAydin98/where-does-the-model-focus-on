from preprocessing import *

# Set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load train and validation files
train_files = os.listdir(config.train_path)
np.random.shuffle(train_files)

val_files = os.listdir(config.val_path)

train_dataset = FruitImages(train_files, transform=train_tf, device=device)
val_dataset = FruitImages(val_files, transform=val_tf, device=device)

train_dl = DataLoader(train_dataset, batch_size=32, collate_fn=train_dataset.collate_fn)
val_dl = DataLoader(val_dataset, batch_size=32, collate_fn=val_dataset.collate_fn)




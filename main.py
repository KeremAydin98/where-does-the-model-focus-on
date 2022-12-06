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


def train_batch(model, data, optimizer, criterion):

    # Prepare the model for training
    model.train()

    # Extract images and labels from data separately
    imgs, labels, _ = data

    # Forward propagation
    preds = model(imgs)

    optimizer.zero_grad()

    # Loss and accuracy
    loss, acc = criterion(preds, labels)

    # Gradient computation
    loss.backward()

    # Backward propagation
    optimizer.step()

    return loss.item(), acc.item()

@torch.no_grad()
def validate_batch(model, data, criterion):

    # Prepare model for evaluation
    model.eval()

    # Extract images and labels from data
    imgs, labels, _ = data

    # Predictions of the model
    preds = model(imgs)

    # Calculating loss and accuracy
    loss, acc = criterion(preds, labels)

    return loss.item(), acc.item()





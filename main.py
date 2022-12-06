from preprocessing import *
from models import FruitClassifier
from glob import glob

# Set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load train and validation files
train_files = glob(config.train_path)
np.random.shuffle(train_files)

val_files = glob(config.val_path)

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


# Instantiate the model
model = FruitClassifier(id2int=train_dataset.id2int).to(device)
criterion = model.compute_metrics
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 2

# Starts the training
for epoch in range(n_epochs):

    N = len(train_dl)

    for bx, data in enumerate(train_dl):

        loss, acc = train_batch(model, data, optimizer, criterion)

        print(f"Epoch: {epoch}/{n_epochs}, Batch: {bx+1}/{N}, Loss: {loss}, Accuracy: {acc}")

    N = len(val_dl)

    for bx, data in enumerate(val_dl):

        loss, acc = validate_batch(model, data, criterion)

        print(f"Epoch: {epoch}/{n_epochs}, Batch: {bx+1}/{N}, Val_loss: {loss}, Val_accuracy: {acc}")
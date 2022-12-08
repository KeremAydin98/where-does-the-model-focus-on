from preprocessing import *
from models import FruitClassifier
from glob import glob
import config
from tqdm import tqdm


def train_batch(model, data, optimizer, criterion):

    # Prepare the model for training
    model.train()

    # Extract images and labels from data separately
    imgs, labels, _ = data

    imgs = torch.tensor(np.array(imgs))
    labels = torch.tensor(np.array(labels))

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

    imgs = torch.tensor(np.array(imgs))
    labels = torch.tensor(np.array(labels))

    # Predictions of the model
    preds = model(imgs)

    # Calculating loss and accuracy
    loss, acc = criterion(preds, labels)

    return loss.item(), acc.item()

def training():
    # Set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load train and validation files
    train_files = glob(config.train_path)

    id2int = len(train_files)

    val_files = glob(config.val_path)

    train_dataset = FruitImages(train_files, transform=train_tf, device=device)
    val_dataset = FruitImages(val_files, transform=val_tf, device=device)

    train_dl = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=train_dataset.collate_fn, drop_last=True)
    val_dl = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=val_dataset.collate_fn, drop_last=True)

    # Instantiate the model
    model = FruitClassifier(id2int=id2int).to(device)
    criterion = model.compute_metrics
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 10

    # Starts the training
    for epoch in range(n_epochs):

        losses = []
        accs = []

        for data in train_dl:
            loss, acc = train_batch(model, data, optimizer, criterion)
            losses.append(loss)
            accs.append(acc)

        val_losses = []
        val_accs = []

        for data in val_dl:
            loss, acc = validate_batch(model, data, criterion)
            val_losses.append(loss)
            val_accs.append(acc)

        print(f"Epoch: {epoch + 1}/{n_epochs}, Loss: {sum(losses) / len(losses)},",
              f" Val_loss: {sum(val_losses) / len(val_losses)},",
              f" Accuracy: {sum(accs) / len(accs)}, Val_accuracy: {sum(val_accs) / len(val_accs)}")


    """
    Saving the model
    """

    torch.save(model.state_dict(), "models/model.pth")


if __name__ == "__main__":
    training()
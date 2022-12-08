import config
from models import *
from preprocessing import *
from glob import glob
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# Instantiate the model
model = FruitClassifier(10).to(device)
model.load_state_dict(torch.load("models/model.pth", map_location=device))

# Fetching the fourth layer of the model
# and also the first two layers within convBlock – which happens to be the
# Conv2D layer
"""
The star expression is used to unpack containers. In your case it would be equal to as if you pass each list
 element separately to nn.Sequential.
"""
img2fmap = nn.Sequential(*(list(model.model[:4]) + list(model.model[4][:2])))

# Test path
test_files = glob(config.test_path)
# Load test dataset
test_ds = FruitImages(test_files, device=device)


def img2cam(x):

    model.eval()

    # Prediction of the model
    prediction = model(x)

    # Extracted features from the fourth convblock of the model
    activations = img2fmap(x)

    # Argmax of the prediction
    pred = prediction.max(-1)[-1]

    # Start the calculation of the gradient by
    model.zero_grad()

    # Computes the gradient of current tensor w.r.t. graph leaves.
    # As long as you use retain_graph=True in your backward method, you can do backward any time you want
    prediction[0][pred].backward(retain_graph=True)

    # .weight() weight of the given layer
    # .grad() after backward propagation
    # .mean() takes the mean for given dimensions
    pooled_grads = model.model[-7][0].weight.grad.mean((0,2,3))

    # Multiply each activation map with corresponding gradient average
    for i in range(activations.shape[1]):

        activations[:,i,:,:] *= pooled_grads[i]

    # Compute the average of all activation maps
    heatmap = torch.mean(activations, dim=1)[0].cpu().detach()

    return heatmap


def preprocess_for_cam(img):

    return val_tf(img).float().expand(1,3,128,128)


heatmap = img2cam(preprocess_for_cam(test_ds[0][0]))

from models import *
from preprocessing import *
from glob import glob

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
test_files = glob("dataset/predict/*/*.jpeg")
# Load test dataset
test_ds = FruitClassifier(test_files, transform=val_tf, device=device)

print(test_ds[0])

def img2cam(x):

    model.eval()

    prediction = model(x)

    heatmaps = []

    extracted_features = img2fmap(x)

    pred = prediction.max(-1)[-1]

    model.zero_grad()

    prediction[0, pred].backward(retain_graph=True)

    pooled_grads = model.model[-7][1].weight.grad.data.mean((0,2,3))

    for i in range(extracted_features.shape[1]):

        extracted_features[:,i,:,:] *= pooled_grads[i]

    heatmap = torch.mean(extracted_features, dim=1)[0].cpu().detach()

    return heatmap


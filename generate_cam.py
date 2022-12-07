from models import *


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
img2fmap = nn.Sequential(*(list(model.model[:5]) + list(model.model[4][:2])))


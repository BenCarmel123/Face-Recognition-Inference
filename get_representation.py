# get_representation.py
import torch
from torchvision import transforms
from PIL import Image
from models.FECNet import FECNet
import numpy as np

DEVICE = torch.device("cpu")  # or "cuda" if available

# Preprocess function
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # normalize pixel values in each channel
                         std=[0.229, 0.224, 0.225])
])

# Load FECNet model once
model = FECNet(pretrained=False)
model.load_state_dict(torch.load("FECNet.pt", map_location=DEVICE), strict=False)
model.to(DEVICE)
model.eval()

"""def get_FECNet_representation(img): # returns vector representing image
    img = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img)
    return output.squeeze().cpu().numpy()"""

def get_FECNet_representation(img): # returns vector representing image
    img = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        # Get the output from the model
        output = model(img)
        # Check if output contains NaN values
        if torch.isnan(output).any():
            print("WARNING: Output contains NaN values!")
    return output.squeeze().cpu().numpy()


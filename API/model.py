import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

#THE CNN Architecture

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024*3*3,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64,out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32,out_features=3)
        )
    def forward(self,x):
        x=self.features(x)
        x=self.classifier(x)
        return x

    
def load_model():
    model = CNN()
    model.load_state_dict(torch.load("potato_model.pth", map_location="cpu"))
    model.eval()
    return model

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

#prediction
def predict(model, image):
    img = Image.open(image).convert("RGB")
    img = transform(img).unsqueeze(0)

    outputs = model(img)
    _, predicted = torch.max(outputs, 1)

    classes = ["Early Blight", "Late Blight", "Healthy"]
    return classes[predicted.item()]

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

#THE CNN Architecture
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
# Device configuration

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
def load_model():
    model = CNN(3)
    model.load_state_dict(torch.load("agriculture_model.pth", map_location=device))
    model.eval()
    return model

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize all images to same size
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#prediction
def predict(model, image):
    img = Image.open(image).convert("RGB")
    img = transform(img).unsqueeze(0)

    outputs = model(img)
    _, predicted = torch.max(outputs, 1)

    classes = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
    return classes[predicted.item()]

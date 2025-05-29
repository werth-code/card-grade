import os
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn

# Constants
MODEL_PATH = 'card_grader/model/model.pth'
DATA_DIR = 'card_grader/dataset/val'  # This is likely where your val data should be
BATCH_SIZE = 32

# Check if validation directory exists
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Validation directory not found: {DATA_DIR}. Please run the full pipeline or check your split path.")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load validation data
dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Validation
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100 if total > 0 else 0
print(f"\n🎯 Validation Accuracy: {accuracy:.2f}%")

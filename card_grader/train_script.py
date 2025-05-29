import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# === CONFIGURATION ===
DATA_DIR = "card_grader/image_process/sorted"
MODEL_PATH = "card_grader/model.pth"
BATCH_SIZE = 16
EPOCHS = 5
IMG_SIZE = 224

# === DATA TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === LOAD DATA ===
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === SETUP MODEL ===
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === TRAINING LOOP ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n🎯 Starting training...")
for epoch in range(EPOCHS):
    running_loss = 0.0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(data_loader):.4f}")

# === SAVE MODEL ===
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n💾 Model saved to {MODEL_PATH}")

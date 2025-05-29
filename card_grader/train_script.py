import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# === CONFIGURATION ===
CONFIG = {
    "data_dir": "card_grader/image_process/sorted",
    "model_output": "card_grader/model/model.pth",
    "batch_size": 16,
    "epochs": 5,
    "learning_rate": 1e-4,
    "image_size": 224,
    "num_classes": 10,
    "use_gpu": torch.cuda.is_available(),
    "shuffle": True,
}

# === DATA TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# === DATASET AND DATALOADER ===
dataset = datasets.ImageFolder(CONFIG["data_dir"], transform=transform)
dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=CONFIG["shuffle"])

# === MODEL ===
device = torch.device("cuda" if CONFIG["use_gpu"] else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, CONFIG["num_classes"])
model = model.to(device)

# === LOSS AND OPTIMIZER ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

# === TRAINING LOOP ===
print("\n🚀 Starting training...")
model.train()
for epoch in range(CONFIG["epochs"]):
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Loss: {avg_loss:.4f}")

# === SAVE MODEL ===
os.makedirs(os.path.dirname(CONFIG["model_output"]), exist_ok=True)
torch.save(model.state_dict(), CONFIG["model_output"])
print(f"\n✅ Model saved to {CONFIG['model_output']}")

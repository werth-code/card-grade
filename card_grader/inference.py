import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "card_grader/model/model.pth"
CLASS_NAMES = [str(i) for i in range(1, 11)]  # Grades 1 through 10

def load_model():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def predict(image_path):
    image_tensor = preprocess_image(image_path)
    model = load_model()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = CLASS_NAMES[predicted.item()]
        print(f"🧠 Predicted Grade: {predicted_class}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to input image")
    args = parser.parse_args()
    predict(args.image_path)

if __name__ == "__main__":
    main()

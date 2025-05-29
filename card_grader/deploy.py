# card_grader/deploy.py
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# === CONFIGURATION ===
MODEL_PATH = "card_grader/model/model.pth"
CLASS_NAMES = [str(i) for i in range(1, 11)]  # Grades 1 through 10
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_grade(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = TRANSFORM(image).unsqueeze(0)

    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        output = model(image_tensor)
        predicted_index = output.argmax(1).item()
        predicted_grade = CLASS_NAMES[predicted_index]

    return predicted_grade

if __name__ == "__main__":
    test_image = input("🖼️ Enter the path to an image to predict the grade: ").strip()
    if not os.path.exists(test_image):
        print("❌ Image not found.")
    else:
        grade = predict_grade(test_image)
        print(f"🎓 Predicted Grade: {grade}")

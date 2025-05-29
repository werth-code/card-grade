import os
import shutil
from pathlib import Path

# === CONFIGURATION ===
SORTED_DIR = "card_grader/image_process/sorted"
TRAIN_DIR = "card_grader/training_data/train"
VAL_DIR = "card_grader/training_data/val"
SPLIT_RATIO = 0.8  # 80% training, 20% validation

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

def prepare_datasets():
    for grade in os.listdir(SORTED_DIR):
        grade_dir = os.path.join(SORTED_DIR, grade)
        if not os.path.isdir(grade_dir):
            continue

        images = [f for f in os.listdir(grade_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total = len(images)
        train_count = int(SPLIT_RATIO * total)

        train_images = images[:train_count]
        val_images = images[train_count:]

        # Create subfolders for the grade class
        os.makedirs(os.path.join(TRAIN_DIR, grade), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, grade), exist_ok=True)

        for img in train_images:
            shutil.copy2(os.path.join(grade_dir, img), os.path.join(TRAIN_DIR, grade, img))
        for img in val_images:
            shutil.copy2(os.path.join(grade_dir, img), os.path.join(VAL_DIR, grade, img))

        print(f"📁 Grade {grade}: {len(train_images)} train / {len(val_images)} val")

if __name__ == "__main__":
    prepare_datasets()
    print("✅ Dataset preparation complete.")

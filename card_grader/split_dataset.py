import os
import shutil
import random
from pathlib import Path

# === CONFIGURATION ===
SOURCE_DIR = "card_grader/image_process/sorted"
DEST_DIR = "card_grader/dataset"
TRAIN_RATIO = 0.8
SEED = 42

def create_split():
    random.seed(SEED)

    train_dir = os.path.join(DEST_DIR, "train")
    val_dir = os.path.join(DEST_DIR, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for grade in os.listdir(SOURCE_DIR):
        grade_dir = os.path.join(SOURCE_DIR, grade)
        if not os.path.isdir(grade_dir):
            continue

        images = list(Path(grade_dir).glob("*.jpg"))
        random.shuffle(images)

        split_idx = int(len(images) * TRAIN_RATIO)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        os.makedirs(os.path.join(train_dir, grade), exist_ok=True)
        os.makedirs(os.path.join(val_dir, grade), exist_ok=True)

        for img in train_images:
            shutil.copy(img, os.path.join(train_dir, grade, img.name))
        for img in val_images:
            shutil.copy(img, os.path.join(val_dir, grade, img.name))

    print("✅ Dataset split complete:")
    print(f"- Train directory: {train_dir}")
    print(f"- Validation directory: {val_dir}")

if __name__ == "__main__":
    create_split()

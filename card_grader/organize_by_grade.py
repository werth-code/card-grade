import os
import shutil
import pandas as pd

# === CONFIGURATION ===
METADATA_FILE = "card_grader/image_process/processed_metadata.csv"
CROP_DIR = "card_grader/image_process/cropped"
SORTED_DIR = "card_grader/image_process/sorted"

os.makedirs(SORTED_DIR, exist_ok=True)

def organize_by_grade():
    if not os.path.exists(METADATA_FILE):
        print(f"❌ Metadata file not found at {METADATA_FILE}")
        return

    df = pd.read_csv(METADATA_FILE)

    for _, row in df.iterrows():
        src_path = row['filepath']
        grade = str(row['grade'])
        filename = os.path.basename(src_path)

        grade_dir = os.path.join(SORTED_DIR, grade)
        os.makedirs(grade_dir, exist_ok=True)

        dest_path = os.path.join(grade_dir, filename)

        try:
            shutil.copy(src_path, dest_path)
            print(f"📁 Moved {filename} to {grade}/")
        except Exception as e:
            print(f"❌ Failed to move {filename}: {e}")

    print(f"\n✅ All images organized into {SORTED_DIR}/<grade>/ folders")

if __name__ == "__main__":
    organize_by_grade()

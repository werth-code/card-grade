import os
import pandas as pd
import shutil

METADATA_FILE = "card_grader/image_process/processed_metadata.csv"
SORTED_DIR = "card_grader/image_process/sorted"

os.makedirs(SORTED_DIR, exist_ok=True)

def sort_images_by_grade():
    df = pd.read_csv(METADATA_FILE)

    for _, row in df.iterrows():
        grade = str(row['grade'])
        filepath = row['filepath']

        if not os.path.exists(filepath):
            print(f"⚠️ Missing file: {filepath}")
            continue

        grade_dir = os.path.join(SORTED_DIR, grade)
        os.makedirs(grade_dir, exist_ok=True)

        filename = os.path.basename(filepath)
        dest_path = os.path.join(grade_dir, filename)

        shutil.copy(filepath, dest_path)

    print(f"✅ Images sorted into {SORTED_DIR}/[grade]/ folders")

if __name__ == "__main__":
    sort_images_by_grade()

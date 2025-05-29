import os
import cv2
import pandas as pd
import numpy as np

# === CONFIGURATION ===
METADATA_FILE = "card_grader/image_process/metadata.csv"
CROP_DIR = "card_grader/image_process/cropped"
DEBUG_DIR = "card_grader/image_process/debug"
OUTPUT_METADATA = "card_grader/image_process/processed_metadata.csv"

os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# Expected card aspect ratio (width:height)
EXPECTED_ASPECT_RATIO = 2.5 / 3.5
ASPECT_RATIO_TOLERANCE = 0.2  # 20% margin
MIN_CARD_AREA_RATIO = 0.1  # Must be at least 10% of image area


def is_valid_card(contour, image_shape):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h != 0 else 0
    area = w * h
    image_area = image_shape[0] * image_shape[1]

    expected_min = EXPECTED_ASPECT_RATIO * (1 - ASPECT_RATIO_TOLERANCE)
    expected_max = EXPECTED_ASPECT_RATIO * (1 + ASPECT_RATIO_TOLERANCE)

    return expected_min <= aspect_ratio <= expected_max and (area / image_area) >= MIN_CARD_AREA_RATIO


def detect_and_crop_card(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to read {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 120)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = [c for c in contours if is_valid_card(c, image.shape)]
    
    if not valid_contours:
        return None

    largest = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    # Save debug image with contour
    debug_img = image.copy()
    cv2.drawContours(debug_img, [largest], -1, (0, 255, 0), 2)
    debug_path = os.path.join(DEBUG_DIR, os.path.basename(image_path).replace(".jpg", "_contours.jpg"))
    cv2.imwrite(debug_path, debug_img)

    return image[y:y+h, x:x+w]


def preprocess():
    if not os.path.exists(METADATA_FILE):
        print(f"❌ Metadata file not found at {METADATA_FILE}")
        return

    metadata = pd.read_csv(METADATA_FILE)
    processed = []

    for _, row in metadata.iterrows():
        input_path = row['filepath']
        grade = row['grade']
        title = row['title']
        url = row['url']

        cropped_image = detect_and_crop_card(input_path)

        if cropped_image is not None:
            crop_filename = os.path.basename(input_path)
            crop_path = os.path.join(CROP_DIR, crop_filename)
            cv2.imwrite(crop_path, cropped_image)
            processed.append([crop_path, grade, title, url])
        else:
            print(f"⚠️ No card detected in {input_path}")

    pd.DataFrame(processed, columns=["filepath", "grade", "title", "url"]).to_csv(OUTPUT_METADATA, index=False)
    print(f"\n📝 Saved processed metadata to {OUTPUT_METADATA}")


if __name__ == "__main__":
    preprocess()

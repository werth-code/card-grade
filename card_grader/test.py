import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from card_grader.image_process.preprocess import detect_and_crop_card
import cv2

test_path = "card_grader/image_process/images/10_2020_Pokemon_Darkness_Ablaze_194_Full_Art_Salamenc.jpg"
output_path = "card_grader/image_process/debug/test_cropped.jpg"

result = detect_and_crop_card(test_path)
if result is not None:
    cv2.imwrite(output_path, result)
    print(f"✅ Cropped image saved to {output_path}")
else:
    print("❌ No card detected.")

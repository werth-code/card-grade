#!/bin/bash

set -e

# Step 1: Optionally scrape and preprocess
read -p "Do you want to scrape and preprocess data? (y/n): " answer
if [[ $answer == "y" ]]; then
  echo "🔍 Running web scrape..."
  python card_grader/websearch.py

  echo "⏳ Preprocessing images (this may take a while)..."
  python card_grader/image_process/preprocess.py

  echo "📂 Sorting images by grade..."
  python card_grader/image_process/sort_by_grade.py

  echo "🔀 Splitting dataset into train/val..."
  python card_grader/split_dataset.py
  echo "✅ Dataset split complete:"
  echo "- Train directory: card_grader/dataset/train"
  echo "- Validation directory: card_grader/dataset/val"
fi

# Step 2: Train model with animated loading dots
animate_training() {
  i=0
  sp=". .. ... .. ."
  while kill -0 $1 2>/dev/null; do
    printf "\r🚀 Starting training%s" "${sp:i++%${#sp}:3}"
    sleep 0.5
  done
  printf "\r✅ Training complete!          \n"
}

python card_grader/train_script.py &
train_pid=$!
animate_training $train_pid
wait $train_pid

# Step 3: Validate model
echo "✅ Validating model..."
python card_grader/validate.py

echo "🎉 Pipeline complete!"

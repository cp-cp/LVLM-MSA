#!/bin/bash

export CUDA_VISIBLE_DEVICES=0


IMAGE_FOLDER="./data/Sarcasm-Detection/MMSD1.0/images/dataset_image_test"
QUESTION_FILE="./results/Baseline/MSD/prompt/baseline.json"
OUTPUT_FOLDER="./results/Baseline/MSD/output/InstuctBLIP/json/baseline.json"

python  ./eval_script/InstuctBLIP/eval_scripts/instructblip_test.py \
    --img_path "$IMAGE_FOLDER" \
    --question "$QUESTION_FILE" \
    --output_folder "$OUTPUT_FOLDER" \
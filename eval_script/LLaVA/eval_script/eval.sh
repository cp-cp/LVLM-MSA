#!/bin/bash

export PYTHONPATH=$PYTHONPATH:~/LLaVA ## change to your own path for LLaVA
 
MODEL_PATH="~/pretrained_models/llava-v1.5-7b" ## change to your own path for ckpts

IMAGE_FOLDER="./data/Sarcasm-Detection/MMSD1.0/images/dataset_image_test"
QUESTION_DIR="./results/Baseline/MSD/prompt/baseline.json"
OUTPUT_FOLDER="./results/Baseline/MSD/output/LLaVA/json/baseline.json"


python ~/LLaVA/llava/eval/model_vqa.py \
    --model-path "$MODEL_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --output-folder "$OUTPUT_FOLDER" 

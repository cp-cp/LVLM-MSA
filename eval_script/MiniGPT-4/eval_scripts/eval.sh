export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:~/MiniGPT-4 ## change to your own path for LLaVA

PORT=12348
DATASET='sd' # sd

IMAGE_FOLDER="./data/Sarcasm-Detection/MMSD1.0/images/dataset_image_test"
EVAL_FILE_PATH="./results/Baseline/MSD/prompt/baseline.json"
ANSWERS_FILE="./results/Baseline/MSD/output/LLaVA/json/baseline.json"

CFG_PATH="~/MiniGPT-4/eval_configs/minigptv2_benchmark_evaluation.yaml" ## change to your own path for ckpts

MAX_TOKEN=1024

torchrun --master-port ${PORT} --nproc_per_node 1 eval_vqa_v2.py \
        --answers-file ${ANSWERS_FILE}\
        --cfg-path ${CFG_PATH} \
        --dataset ${DATASET}\
        --img-path ${IMAGE_FOLDER}\
        --eval-file-path ${EVAL_FILE_PATH}\
        --batch-size 2\
        --max_new_tokens $MAX_TOKEN\

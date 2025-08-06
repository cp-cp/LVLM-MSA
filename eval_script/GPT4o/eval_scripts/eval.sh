
json_file="./results/Baseline/MSD/prompt/baseline.json"
image_folder="./data/Sarcasm-Detection/MMSD1.0/images/dataset_image_test"
api_key="<put your api key here>"
output_file="./results/Baseline/MSD/output/GPT4o/json/baseline.json"
python model_vqa.py --json_file $json_file\
    --image_folder $image_folder\
    --api_key $api_key \
    --output_file $output_file\

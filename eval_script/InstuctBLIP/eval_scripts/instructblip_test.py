import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from tqdm import tqdm
import argparse
import os
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--img_path", type=str, default="./", help="Path to the images directory.")
    parser.add_argument("--output_folder", type=str, default="", help="Path to the output directory.")
    parser.add_argument("--question", type=str, default="./", help="Path to the questions JSONL file.")
    parser.add_argument("--question_dir", type=str, default="", help="Path to the questions directory.")
    parser.add_argument("--answer", type=str, default="./answer.jsonl", help="Path to save the answers JSONL file.")
    parser.add_argument("--is_7b", type=bool, default=False, help="Flag to load the 7B model.")
    parser.add_argument("--options", nargs="+", help="Override some settings in the used config.")
    
    return parser.parse_args()

def get_answer(args,sample, vis_processors, model, device):
    image_file = sample["image"]

    image_file = os.path.join(args.img_path, sample["image"])
    raw_image = Image.open(image_file).convert("RGB")

    # Prepare the image
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    prompt = sample["instruction"]
    prompt = prompt.encode('utf-8', 'ignore').decode('utf-8')

    print(prompt)
    result=model.generate({"image": image, "prompt": prompt})
    return result[0]

def main():
    args = parse_args()
    
    question_file_base = os.path.splitext(os.path.basename(args.question))[0]
    json_output_folder = os.path.join(os.path.expanduser(args.output_folder), "json")
    txt_output_folder = os.path.join(os.path.expanduser(args.output_folder), "txt")
    answers_file = os.path.join(json_output_folder, question_file_base + "_ans.jsonl")
    txt_answers_file = os.path.join(txt_output_folder, question_file_base + "_ans.txt")

    os.makedirs(json_output_folder, exist_ok=True)
    os.makedirs(txt_output_folder, exist_ok=True)
    
    ans_file = open(answers_file, "w")
    txt_ans_file = open(txt_answers_file, "w")

    # Setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
    # Load InstructBLIP model
    model_name = "blip2_vicuna_instruct"
    model_type = "vicuna7b" #if args.is_7b else "vicuna13b"
    model, vis_processors, _ = load_model_and_preprocess(name=model_name, model_type=model_type, is_eval=True, device=device)

    # Load test set
    test_set = []
    with open(args.question, "r") as f:
        for line in f:
            test_set.append(eval(line.strip()))

    instruct_blip_answer = []

    for sample in tqdm(test_set):
        sample["id"] = sample["image"]
        if "question" in sample:
            sample["instruction"] = sample["question"]
        elif "text" in sample:
            sample["instruction"] = sample["text"]

        answer = {
            "question_id": sample["id"],
            "image": sample["image"],
            "prompt": sample["instruction"],
            "text": get_answer(args,sample, vis_processors, model, device)
        }
        # instruct_blip_answer.append(answer)
        ans_file.write(json.dumps(answer) + "\n")
        ans_file.flush()

        txt_ans_file.write(answer["text"] + "\n")
        txt_ans_file.flush()

    # with open(args.answer, "w") as f:
    # for ll in instruct_blip_answer:
    #     ans_file.write(json.dumps(ll) + "\n")
    #     ans_file.flush()

    #     txt_ans_file.write(ll["text"] + "\n")
    #     txt_ans_file.flush()

    ans_file.close()
    txt_ans_file.close()

if __name__ == "__main__":
    main()

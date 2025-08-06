import os
import re
import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from minigpt4.datasets.datasets.vqa_datasets import (
    OKVQAEvalData, VizWizEvalData, IconQAEvalData, GQAEvalData, 
    VSREvalData, HMEvalData, SDEvalData, EMEvalData,EvalData
)
from minigpt4.common.vqa_tools.VQA.PythonHelperTools.vqaTools.vqa import VQA
from minigpt4.common.vqa_tools.VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval

from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.conversation.conversation import CONV_VISION_minigptv2

def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default=['refcoco'], help="dataset to evaluate")
parser.add_argument("--answers-file", type=str, default='../output.json')
parser.add_argument("--eval-file-path", type=str, help="Path to the evaluation file")
parser.add_argument("--img-path", type=str, help="Path to the images")
parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation")
parser.add_argument("--max-new-tokens", type=int, default=50, help="Maximum new tokens to generate")
args = parser.parse_args()

model, vis_processor = init_model(args)
conv_temp = CONV_VISION_minigptv2.copy()
conv_temp.system = ""
model.eval()
save_path = args.answers_file

def evaluate_model(dataset_type, DataClass=EvalData):
    annotation = []
    with open(args.eval_file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            json_obj = json.loads(line)
            annotation.append(json_obj)

    data = DataClass(annotation, vis_processor, args.img_path)
    eval_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    count = 0
    total = 0

    minigpt4_predict = []

    

    with open(save_path, 'w') as f:
        for images, texts in tqdm(eval_dataloader):
            texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
            answers = model.generate(images, texts, max_new_tokens=args.max_new_tokens, do_sample=False)

            for answer in answers:
                result = dict()
                if dataset_type == 'hm' or dataset_type == 'sd':
                    if answer.lower().strip() in ["yes", "A"]:
                        answer = "A"
                    elif answer.lower().strip() in ["no", "B"]:
                        answer = "B"
                    else:
                        print("non-matching answer", answer)
                elif dataset_type == 'em':
                    if answer.lower().strip() in ["positive", "negative", "neutral"]:
                        answer = answer.lower().strip()
                    else:
                        print("non-matching answer", answer)
                else:
                    print("Unsupported dataset type:", dataset_type)
                    
                result['pred'] = answer
                minigpt4_predict.append(result)
                f.write(json.dumps(result)+"\n")
                f.flush()

for dataset in args.dataset:
    if dataset == 'hm':
        # evaluate_model('hm', HMEvalData)
        evaluate_model('hm')
    elif dataset == 'sd':
        # evaluate_model('sd', SDEvalData)
        evaluate_model('sd')
    elif dataset == 'em':
        # evaluate_model('em', EMEvalData)
        evaluate_model('em')
    else:
        print(f"Dataset {dataset} not supported.")

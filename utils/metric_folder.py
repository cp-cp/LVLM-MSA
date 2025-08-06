import os
import json
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
import nltk
from rouge_score import rouge_scorer
from pprint import pprint
import torch

def open_file(hyp_file, gt_file):
    with open(hyp_file, 'r', encoding='utf-8') as r1, open(gt_file, 'r', encoding='utf-8') as r2:
        for line1, line2 in zip(r1, r2):
            a, b, c = 'empty', line1.strip(), line2.strip()
            yield a, b, c

def eval_metrics(hyp_txt, ref_txt):
    # calculate metrics
    with open(ref_txt, 'r') as gtr:
        gt_list = gtr.readlines()
    with open(hyp_txt, 'r') as genr:
        gen_list = genr.readlines()
        
    print(len(gt_list), len(gen_list))

    for_eval_gt_dic = {}
    for_eval_gen_dic = {}
    for_dist_hyper = []

    for i in range(len(gen_list)):

        ref = ' '.join(nltk.word_tokenize(gt_list[i].lower()))
        gen = ' '.join(nltk.word_tokenize(gen_list[i].lower()))
        for_dist_hyper.append(nltk.word_tokenize(gen_list[i].lower()))

        for_eval_gt_dic[i] = [ref]
        for_eval_gen_dic[i] = [gen]

    scores = eval_metrics_str(for_eval_gen_dic, for_eval_gt_dic)

    dist_dic = calc_distinct(for_dist_hyper)
    scores.update(dist_dic)

    for k in scores:
        scores[k] = f"{scores[k] * 100:.3f}"
    return scores

def eval_metrics_str(hyp_dic, ref_dic):
    hyp_list = []
    ref_list = []
    for i in hyp_dic.values():
        hyp_list.append(i)
    for i in ref_dic.values():
        ref_list.append(i)
    result_dic = {}
    b = Bleu()
    score, _ = b.compute_score(gts=ref_dic, res=hyp_dic)
    b1, b2, b3, b4 = score

    r = Rouge()
    score, _ = r.compute_score(gts=ref_dic, res=hyp_dic)
    rl = score

    count = 0
    rouge1 = 0
    rouge2 = 0
    met = 0
    for reference, hypothesis in zip(ref_list, hyp_list):
        reference = str(reference).strip('[]\'')
        hypothesis = str(hypothesis).strip('[]\'')
        count += 1
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        met += nltk.translate.meteor_score.meteor_score([nltk.word_tokenize(reference)], nltk.word_tokenize(hypothesis))  # 分词处理
        scores = scorer.score(reference, hypothesis)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
    rouge1 = rouge1 / count
    rouge2 = rouge2 / count
    met = met / count
    c = Cider()
    score, _ = c.compute_score(gts=ref_dic, res=hyp_dic)
    cdr = score

    result_dic['Bleu_1'] = b1
    result_dic['Bleu_2'] = b2
    result_dic['Bleu_3'] = b3
    result_dic['Bleu_4'] = b4
    result_dic['Rouge_L'] = rl
    result_dic['Rouge1'] = rouge1
    result_dic['Rouge2'] = rouge2
    result_dic['METEOR'] = met
    return result_dic

def calc_distinct(hyps):
    dist_dic = {}
    for k in range(1, 3):
        d = {}
        tot = 0
        for sen in hyps:
            for i in range(0, len(sen) - k):
                key = tuple(sen[i:i + k])
                d[key] = 1
                tot += 1
        if tot > 0:
            dist = len(d) / tot
        else:
            warnings.warn('the distinct is invalid')
            dist = 0.
        dist_dic[f'Dist-{k}'] = dist
    return dist_dic

def evaluate_directory(txt_dir, ref_file, output_json):
    # model = ["LLaVA","MiniGPTv2","InstructBLIP"]
    results = {}
    # for file_name in os.listdir(txt_dir):
    #     if file_name.endswith(".txt"):
    #         file_path = os.path.join(txt_dir, file_name)
    #         print(f"Evaluating {file_name}")
    #         scores = eval_metrics(file_path, ref_file)
    #         results[file_name] = scores

    for root, dirs, files in os.walk(txt_dir):
        for file_name in files:
            if file_name.endswith(".txt") and file_name.endswith("SE_gt.txt") == False:
                file_path = os.path.join(root, file_name)
                print(f"Evaluating {file_path}")
                scores = eval_metrics(file_path, ref_file)
                results[file_path] = scores

    with open(output_json, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # 设定要评估的文件夹路径和ground truth文件路径
    txt_dir = './Baseline/MSE/output/MiniGPTv2-2'
    ref_file = './Baseline/MSE/output/SE_gt.txt'
    output_json = txt_dir+"evaluation_results.json"

    evaluate_directory(txt_dir, ref_file, output_json)

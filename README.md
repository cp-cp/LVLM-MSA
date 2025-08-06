# LVLM-MSA

[![arxiv](https://img.shields.io/badge/arXiv-2508.03654-b31b1b.svg)](https://arxiv.org/abs/2508.03654)

This repo holds codes of the paper: ''Can Large Vision-Language Models Understand Multimodal Sarcasm?'' accepted by the conference CIKM 2025. 


## 📂 Dataset Preparation

We evaluate LVLMs on Multimodal Sarcasm Analysis (MSA) in two tasks:

- **Multimodal Sarcasm Detection (MSD)**  
  Dataset: [MSDD](https://github.com/headacheboy/data-of-multimodal-sarcasm-detection)

- **Multimodal Sarcasm Explanation (MSE)**  
  Dataset: [MORE](https://github.com/LCS2-IIITD/Multimodal-Sarcasm-Explanation-MuSE/tree/main)

Download and prepare these datasets according to the instructions in their respective repositories.


## 🔍 Fine-Grained Object Extraction

We follow the [py-bottom-up-attention](https://github.com/airsplay/py-bottom-up-attention) implementation to extract fine-grained objects along with their descriptors.

1. Install the required environment following the instructions in the linked repository.
2. Run object extraction:

```shell
cd py-bottom-up-attention
python test.py
```

## 🌐 External Knowledge Acquisition

After extracting objects, we retrieve related conceptual knowledge from ConceptNet to enhance sarcasm understanding.


```shell
cd obj_concept
python obj_concept.py
```
This step links entities to relevant commonsense and sentiment-related concepts.

## 🧪 Experiment

Our experiments are conducted using **GPT-4o** (via API inference), [InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip), [LLaVA](https://github.com/haotian-liu/LLaVA), and [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4).  
Detailed model configurations can be found in **Section 4.2** of our paper.

The evaluation scripts are available in the `./eval_script` directory.  
Please note that you need to set up each model according to the instructions provided in their **official repositories** before running the evaluation.


## 📖 Citation

If you find this repo useful in your research works, please consider citing:

```
@misc{wang2025largevisionlanguagemodelsunderstand,
      title={Can Large Vision-Language Models Understand Multimodal Sarcasm?}, 
      author={Xinyu Wang and Yue Zhang and Liqiang Jing},
      year={2025},
      eprint={2508.03654},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.03654}, 
}
```
import os
import json

def caculate(predicted_labels, true_labels):
    id_list = []

    for i, (true_label, predicted_label) in enumerate(zip(true_labels, predicted_labels)):
        if true_label != predicted_label:
            id_list.append(i + 1)

    # 计算TP, FP, TN, FN
    TP = sum((true_label == 'A\n') and (predicted_label == 'A\n') for true_label, predicted_label in zip(true_labels, predicted_labels))
    FP = sum((true_label == 'A\n') and (predicted_label == 'B\n') for true_label, predicted_label in zip(true_labels, predicted_labels))
    TN = sum((true_label == 'B\n') and (predicted_label == 'B\n') for true_label, predicted_label in zip(true_labels, predicted_labels))
    FN = sum((true_label == 'B\n') and (predicted_label == 'A\n') for true_label, predicted_label in zip(true_labels, predicted_labels))

    # 计算精确率
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "True Positives (TP)": TP,
        "False Positives (FP)": FP,
        "True Negatives (TN)": TN,
        "False Negatives (FN)": FN,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score
    }

def process_folder(pred_folder, true_file, result_file):
    # Read true labels
    with open(true_file, 'r') as file:
        true_labels = [line for line in file]

    results = []

    # Process each file in the folder
    for filename in os.listdir(pred_folder):
        if filename.endswith('.txt'):
            pred_file = os.path.join(pred_folder, filename)

            with open(pred_file, 'r') as file:
                predicted_labels = [line for line in file]

            # Ensure the number of labels matches
            if len(predicted_labels) != len(true_labels):
                raise ValueError(f"Number of lines in {pred_file} does not match the number of lines in {true_file}")

            metrics = caculate(predicted_labels, true_labels)
            results.append({
                "File": filename,
                **metrics
            })

    # Write results to file
    with open(result_file, 'w') as file:
        json.dump(results, file, indent=4)

    print(f"Results saved to {result_file}")

if __name__ == "__main__":
    model='MiniGPTv2'
    task="MSD"
    # model='InstructBLIP'
    pred_folder = f'''./Baseline/{task}/output/{model}/txt'''
    true_file = f'''./data/Sarcasm-Detection/MMSD1.0/SD_gth.txt'''
    result_file = pred_folder+'/results.json'

    process_folder(pred_folder, true_file, result_file)

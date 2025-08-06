import json


def caculate(predicted_labels,true_labels):

    id_list=[]
    # print(zip(true_labels,predicted_labels))

    for i,(true_label, predicted_label) in enumerate(zip(true_labels, predicted_labels)):
        if true_label != predicted_label :
            id_list.append(i+1)


    # 计算TP, FP, TN, FN
    TP = sum((true_label == 'A\n') and (predicted_label == 'A\n') for true_label, predicted_label in zip(true_labels, predicted_labels))
    FP = sum((true_label == 'A\n') and (predicted_label != 'A\n') for true_label, predicted_label in zip(true_labels, predicted_labels))
    TN = sum((true_label == 'B\n') and (predicted_label == 'B\n') for true_label, predicted_label in zip(true_labels, predicted_labels))
    FN = sum((true_label == 'B\n') and (predicted_label != 'B\n') for true_label, predicted_label in zip(true_labels, predicted_labels))

    # 打印结果
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Negatives (FN): {FN}")

    # 计算精确率
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP) > 0 else 0
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # 计算召回率
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # 计算F'A\n'分数
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 打印结果
    print(f"Accuracy: {accuracy:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Recall: {recall:.5f}")
    print(f"F1 Score: {f1_score:.5f}")

    # print((id_list))


if __name__ == "__main__":

    ans_llava_7b=[]
    ans_llava_13b=[]
    # true_file = './data/hateful_memes/gth.txt'    
    pred_file = './Baseline/MSD/output/MiniGPTv2-2/txt/baseline.txt'
    true_file = './Obj/MSD/output/SD_gth.txt'

    with open(pred_file, 'r') as file:
        for line in file:
            ans_llava_7b.append(line)
    check_list=[]
    with open(true_file, 'r') as file:
        for line in file:
            check_list.append(line)

    # 计算check_list中的正样本数
    positive_samples = sum((label) == 'A\n' for label in check_list)
    negative_samples = sum((label) == 'B\n' for label in check_list)
    print(f"Positive Samples: {positive_samples}")
    print(f"Negative Samples: {negative_samples}")


    true_labels = [(label) for label in check_list]
    predicted_labels = [(label) for label in ans_llava_7b]
    # print("LLaVA-v1.5-7B Rational Combine:")
    caculate(predicted_labels,true_labels)

    # predicted_labels = [(label) for label in ans_llava_13b]
    # print("LLaVA-v1.5-13B:")
    # caculate(predicted_labels,true_labels)
import csv
import ast
import json

color_word = [
    'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'white', 'gray', 'cyan', 'magenta',
    'maroon', 'navy', 'olive', 'lime', 'teal', 'indigo', 'violet', 'gold', 'silver', 'bronze', 'beige', 'coral', 
    'turquoise', 'peach', 'lavender', 'mustard', 'mint', 'salmon', 'plum', 'rose', 'tan', 'burgundy', 'emerald', 
    'ruby', 'sapphire', 'amber', 'aquamarine', 'fuchsia', 'ivory', 'jade', 'lilac', 'pearl', 'periwinkle', 'scarlet',
    'seafoam', 'slate', 'tangerine', 'topaz'
]
# 读取CSV文件并解析数据
def read_and_parse_csv(file_path):
    data = {}
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # 读取标题行
        for row in csv_reader:
            key = row[0]
            # print(row[1])
            value = ast.literal_eval(row[1])  # 将字符串转换为列表
            data[key] = value
    return data

def read_and_parse_json(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 文件路径
obj_file_test = 'obj_data/test_obj.csv'  
obj_file_train = 'obj_data/train_obj.csv'  
obj_file_valid = 'obj_data/valid_obj.csv'
obj_file_text='obj_data/text_obj.csv'
concept_file = 'knowledge_data/ConceptNet_VAD_dict.json'
output_file = 'obj_concept_text.csv'  # 输出文件路径

# 读取数据
obj_data_test = read_and_parse_csv(obj_file_test)
obj_data_train = read_and_parse_csv(obj_file_train)
obj_data_valid = read_and_parse_csv(obj_file_valid)
obj_data_text = read_and_parse_csv(obj_file_text)
concept_data = read_and_parse_json(concept_file)

# 保存结果到CSV文件
with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['Key', 'Value'])  # 写入标题行
    for key, value in obj_data_text.items():
        row = []
        for obj in value:
            # 拆分出第一个词
            first_word = obj.split()[0]  # 默认使用空格分割，取第一个词
            # second_word = obj.split()[1]  # 默认使用空格分割，取第二个词
            if first_word in concept_data and first_word not in color_word:
                row.append(f'''{first_word} {concept_data.get(first_word)[0][1]} {concept_data.get(first_word)[0][0]}''')
            # if second_word in concept_data and second_word not in color_word:
                # row.append(f'''{second_word} {concept_data.get(second_word)[0][1]} {concept_data.get(second_word)[0][0]}''')
        csv_writer.writerow([key, row])
    # for key, value in obj_data_train.items():
    #     row = []
    #     for obj in value:
    #         # 拆分出第一个词
    #         first_word = obj.split()[0]  # 默认使用空格分割，取第一个词
    #         second_word = obj.split()[1]  # 默认使用空格分割，取第二个词
    #         if first_word in concept_data and first_word not in color_word:
    #             row.append(f'''{first_word} {concept_data.get(first_word)[0][1]} {concept_data.get(first_word)[0][0]}''')
    #         if second_word in concept_data and second_word not in color_word:
    #             row.append(f'''{second_word} {concept_data.get(second_word)[0][1]} {concept_data.get(second_word)[0][0]}''')
    # for key, value in obj_data_valid.items():
    #     row = []
    #     for obj in value:
    #         # 拆分出第一个词
    #         first_word = obj.split()[0]  # 默认使用空格分割，取第一个词
    #         second_word = obj.split()[1]  # 默认使用空格分割，取第二个词
    #         if first_word in concept_data and first_word not in color_word:
    #             row.append(f'''{first_word} {concept_data.get(first_word)[0][1]} {concept_data.get(first_word)[0][0]}''')
    #         if second_word in concept_data and second_word not in color_word:
    #             row.append(f'''{second_word} {concept_data.get(second_word)[0][1]} {concept_data.get(second_word)[0][0]}''')


print(f"结果已保存到 {output_file}")

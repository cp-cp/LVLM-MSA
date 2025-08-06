# 读取JSON文件
import json

input_file = "./Baseline/MSE/output/MiniGPT4/json/baseline.json"
output_file = "./Baseline/MSE/output/MiniGPT4/txt/baseline.txt"

# 读取JSON文件
with open(input_file, 'r', encoding='utf-8') as json_file, open(output_file, 'w', encoding='utf-8') as txt_file:
    for line in json_file:
        # 解析每一行JSON对象
        data = json.loads(line)
        # 提取text字段内容并转义换行符
        text_content = data.get('text', '').replace('\n', '\\n')
        # 将转义后的内容写入TXT文件
        txt_file.write(text_content + '\n')  # 这里的 '\n' 是为了保证每个JSON对象的text字段独立
print(output_file)

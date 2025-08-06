def process_file(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        with open(output_file, 'w', encoding='utf-8') as file:
            for line in lines:
                if 'b. unsarcastic' in line.lower() or 'b. no' in line.lower():
                    # 替换整行内容为 'B'
                    file.write('B\n')
                elif 'a. sarcastic' in line.lower() or 'a. Yes'in line.lower():
                    file.write('A\n')
                elif line.lower()[0:3]=='yes':
                    file.write('A\n')
                elif line.lower()[0:2] =='no':
                    file.write('B\n')
                elif line.lower()[0:1] =='a':
                    file.write('A\n')
                elif line.lower()[0:1] =='b':
                    file.write('B\n')
                elif line.lower()[0:9] =='sarcastic':
                    file.write('A\n')
                elif line.lower()[0:11] =='unsarcastic':
                    file.write('B\n')
                elif line.lower()== "\n":
                    file.write('B\n')
                elif "the text is not sarcastic" in line.lower():
                    file.write('B\n')
                elif "the text is sarcastic" in line.lower():
                    file.write('A\n')
                elif "the text's apparent meaning is sarcastic" in line.lower():
                    file.write('A\n')
                elif "the text's apparent meaning is not sarcastic" in line.lower():
                    file.write('B\n')
                elif "the text's apparent meaning contrasts with what you might expect from the image." in line.lower():
                    file.write('A\n')
                elif "A." in line:
                    file.write('A\n')
                elif "B." in line:
                    file.write('B\n')
                elif "unsarcastic" in line.lower() or "not sarcastic" in line.lower():
                    file.write('B\n')
                elif "sarcastic" in line.lower():
                    file.write('A\n')
                else:
                    # 保持原有内容
                    # file.write(line)
                    file.write('B\n')

        print(f'Processing complete. The updated file is saved as {output_file}')
    except Exception as e:
        print(f'An error occurred: {e}')

# 使用示例
input_file = './Baseline/MSD/output/MiniGPT/sd.txt'  # 输入文件名
output_file = './Baseline/MSD/output/MiniGPT/out/sd.txt'  # 输出文件名
process_file(input_file, output_file)

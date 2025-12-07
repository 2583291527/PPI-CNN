import math
import statistics
import pandas as pd
from collections import Counter

# 读取 Excel 文件中的数据
input_filename =r"C:\Users\25832\Desktop"
df = pd.read_excel(input_filename, header=None)

# 用于存储最终结果
result = []

# 遍历每一行数据
for index in range(len(df)):
    input_string = df.iloc[index, 0]  # 获取第index行的字符串
    # 统计字母出现次数
    letter_count = Counter(input_string)
    total_A = letter_count.get('A', 0)
    total_C = letter_count.get('C', 0)
    total_G = letter_count.get('G', 0)
    total_T = letter_count.get('T', 0)

    # 用于存储计算结果
    result_x = []
    result_y = []

    # 遍历字符串中的每个字母，逐个计算
    for i, letter in enumerate(input_string, 1):
        if letter == 'A':
            count_A = input_string[:i].count('A')
            result_x.append(math.cos(math.pi / 2 * (count_A / (total_A + 1))))
            result_y.append(math.sin(math.pi / 2 * (count_A / (total_A + 1))))
        elif letter == 'C':
            count_C = input_string[:i].count('C')
            result_x.append(math.cos(math.pi / 2 + math.pi / 2 * (count_C / (total_C + 1))))
            result_y.append(math.sin(math.pi / 2 + math.pi / 2 * (count_C / (total_C + 1))))
        elif letter == 'G':
            count_G = input_string[:i].count('G')
            result_x.append(math.cos(math.pi + math.pi / 2 * (count_G / (total_G + 1))))
            result_y.append(math.sin(math.pi + math.pi / 2 * (count_G / (total_G + 1))))
        elif letter == 'T':
            count_T = input_string[:i].count('T')
            result_x.append(math.cos(3 * math.pi / 2 + math.pi / 2 * (count_T / (total_T + 1))))
            result_y.append(math.sin(3 * math.pi / 2 + math.pi / 2 * (count_T / (total_T + 1))))
    try:
        mean_x = statistics.mean(result_x)
        variance_x = statistics.variance(result_x)
    except statistics.StatisticsError:
        mean_x = variance_x = None  # 如果无法计算均值和方差，设置为None

    try:
        mean_y = statistics.mean(result_y)
        variance_y = statistics.variance(result_y)
    except statistics.StatisticsError:
        mean_y = variance_y = None  # 如果无法计算均值和方差，设置为None

    # 将结果存储到列表
    result.append((mean_x, mean_y, variance_x, variance_y))

# 将结果保存为 DataFrame
output_df = pd.DataFrame(result)

# 输出到新的 Excel 文件
output_filename =r"C:\Users\Desktop"
output_df.to_excel(output_filename, index=False, header=False)

# 打印保存成功消息
print(f"结果已保存到 {output_filename}")




import openpyxl
import pandas as pd

def divide_sequence(sequence):
    length = len(sequence)
    num_segments = 4
    segment_length = length // num_segments  # 每段的长度
    remainder = length % num_segments  # 余数

    segments = []
    start = 0
    for i in range(num_segments):
        end = start + segment_length
        if i == num_segments - 1:  # 最后一段
            end += remainder
        segments.append(sequence[start:end])
        start = end

    return segments


def assign_values(segments):
    values = {'A': [], 'C': [], 'D': [], 'E': [], 'F': [], 'G': [], 'H': [], 'I': [], 'K': [], 'L': [],
              'M': [], 'N': [], 'P': [], 'Q': [], 'R': [], 'S': [], 'T': [], 'V': [], 'W': [], 'Y': []}
    for i, segment in enumerate(segments):
        for char in values.keys():
            segment_values = []
            for j, c in enumerate(segment):
                if c == char:
                    segment_values.append(j+1)
                else:
                    segment_values.append(0)
            values[char].extend(segment_values)

    return values

def averge_num(num_list):
    sum_num = sum(num_list)
    length = len(num_list)
    if length == 0:
        average = 0
    else:
        average = sum_num / length
    return average

def frequency_num(List, num_list):
    Length = len(List)
    L = len(num_list)
    frequency = (Length + 1) / L
    return frequency

def sum_length(num_list):
    L = len(num_list)
    sum_length = ((1+L) * L) // 2

    return sum_length


values_list=[]
df = pd.read_excel(, header=None)
df_list = df.values.tolist()
for i in range(len(df_list)):
    sequence = df_list[i][0]
    segments = divide_sequence(sequence)
    values = assign_values(segments)
    values_list.append(list(values.values()))  # 将每个sequence对应的values结果转换为列表，并添加到values_list中


num = 0
new_list = []
d_num_list = []
D_num_list = []
feature_list = []

wb = openpyxl.Workbook()
ws = wb.active

for sublist in values_list:
    for item in sublist:
        segment = divide_sequence(item)  # 数字序列分段
        each_feature_list = []  # 每小段特征序列

        for num_list in segment: # 提取每段数字序列
            for i in range(len(num_list)):
                if num_list[i] > 0:
                    num = num + 1
                    new_list.append(num_list[i])
            if num > 1:
                L = len(new_list)
                while L >= 2:
                    d_num = new_list[L - 1] - new_list[L - 2]
                    d_num_list.append(d_num)  # 变动序列d_num_list
                    L -= 1
            elif num == 1:
                d_num = new_list[0] - sum_length(num_list) / len(num_list)
                d_num_list.append(d_num)
            elif num == 0:
                d_num = 0
                d_num_list.append(d_num)
            D_num_list.append(d_num_list)  # 变动序列分段
            num = 0
            new_list = []
            d_num_list = []
        j = 0
        m = 0
        while j <= 3:
            new_num_list = segment[j]
            i = D_num_list[m]
            max_num = max(i)
            min_num = min(i)
            ave = averge_num(i)
            frequency = frequency_num(i, new_num_list)
            each_feature_list.extend([max_num, min_num, ave, frequency])  # 每小段特征序列

            m = m + 1
            j = j + 1

        feature_list.append(each_feature_list)

        D_num_list = []



row = 1
for sublist in feature_list:
    for col, value in enumerate(sublist):
        ws.cell(row=row, column=col+1).value = value
    row += 1

wb.save()






























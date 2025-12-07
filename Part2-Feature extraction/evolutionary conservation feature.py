import pandas as pd
import numpy as np
import os

# 定义 sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 获取文件夹中所有的 Excel 文件
folder_path = r
files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

# 存储所有文件的结果
all_results = []

# 遍历所有 Excel 文件
for file_name in files:
    # 读取 Excel 文件，跳过第一列（从第二列开始读取），不使用标题行
    file_path = os.path.join(folder_path, file_name)
    data = pd.read_excel(file_path,usecols=range(1, 21))
    df = pd.DataFrame(data)

    # 对数据应用 Sigmoid 函数进行归一化
    df_normalized = df.apply(lambda x: x.apply(sigmoid))

    # 计算列的均值
    column_means = df_normalized.mean()

    L = len(df_normalized)
    a = 3  # 偏移量

    result_matrix = np.zeros((len(df_normalized.columns), a))

    # 遍历每列
    for col_idx, col in enumerate(df_normalized.columns):
        # 遍历每个偏移量，从 1 到 a
        for offset in range(1, a + 1):
            sum_result = 0
            # 对于每个偏移量，计算所有 i 和 i + offset 之间的差异乘积并累加
            for i in range(L - offset):
                xi = df_normalized[col].iloc[i] - column_means[col]
                xi_offset = df_normalized[col].iloc[i + offset] - column_means[col]
                sum_result += (xi * xi_offset) / (L - offset)

            # 将每个偏移量的计算结果存储在结果矩阵中
            result_matrix[col_idx, offset - 1] = sum_result

    # 将结果矩阵转置
    result_matrix_transposed = result_matrix.T

    # 将转置后的矩阵拼接为一行
    result_flattened = result_matrix_transposed.flatten()

    # 将列均值和计算结果合并
    final_result = np.append(column_means.values, result_flattened)

    file_name_without_extension = os.path.splitext(file_name)[0]

    # 添加文件名到结果列表
    all_results.append([file_name_without_extension] + final_result.tolist())

# 将所有结果写入新的 Excel 文件
final_df = pd.DataFrame(all_results)
final_df.to_excel(r, header=False, index=False)








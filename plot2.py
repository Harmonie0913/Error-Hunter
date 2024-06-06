csv_file_path = r'C:\Users\yangx\OneDrive - KUKA AG\Bachelor_Thesis\TCPDATA\feature selection.csv'
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np


# 读取CSV文件
df = pd.read_csv(csv_file_path)

# 获取标签和特征列
labels = df.iloc[:, 0]
features = df.iloc[:, 1:25]

# 检查标签列的唯一值
unique_labels = labels.unique()
print(f"Unique labels: {unique_labels}")

# 生成特征对的组合
feature_pairs = list(itertools.permutations(features.columns, 2))

# 定义颜色字典和标签名字典
color_dict = {unique_labels[0]: 'red', unique_labels[1]: 'blue',unique_labels[2]: 'green'}
label_name_dict = {unique_labels[0]: 'IO',  unique_labels[1]: 'NIO',unique_labels[2]: 'IO with 0'}

# 绘制每对特征的散点图
for (x_feature, y_feature) in feature_pairs:
    plt.figure(figsize=(10, 6))
    for label in unique_labels:
        subset = df[df.iloc[:, 0] == label]
        plt.scatter(subset[x_feature], subset[y_feature], label=label_name_dict[label], color=color_dict[label])
    
    # 新增功能：只显示前两个标签都存在的那一块区域
    subset1 = df[df.iloc[:, 0] == unique_labels[0]]
    subset2 = df[df.iloc[:, 0] == unique_labels[1]]
    min_y = np.maximum(np.min(subset1[y_feature]), np.min(subset2[y_feature]))
    max_y = np.minimum(np.max(subset1[y_feature]), np.max(subset2[y_feature]))
    plt.fill_between([np.min(df[x_feature]), np.max(df[x_feature])], min_y, max_y, color='grey', alpha=0.5)
    
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title(f'Scatter Plot of {x_feature} vs {y_feature}')
    plt.legend()
    plt.show()
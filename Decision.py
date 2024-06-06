csv_file_path = r'C:\Users\yangx\OneDrive - KUKA AG\Bachelor_Thesis\TCPDATA\feature selection.csv'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk
# 假设 csv_file_path 是你的 CSV 文件路径

# 读取数据
df = pd.read_csv(csv_file_path)
# 获取标签和特征列
labels = df.iloc[:, 0]
features = df.iloc[:, 1:25]
# 检查标签列的唯一值
unique_labels = labels.unique()
print(f"Unique labels: {unique_labels}")


# 定义颜色字典和标签名字典
color_dict = {unique_labels[0]: 'red', unique_labels[1]: 'blue'}
label_name_dict = {unique_labels[0]: 'IO',  unique_labels[1]: 'NIO'}
# 设置保存路径
save_path = "all_plots with 2 labels.png"
# 创建大窗口并绘制所有子图
fig, axes = plt.subplots(nrows=100, ncols=6, figsize=(30, 500))
axes = axes.flatten()
index = 0
for i in range(1, 26):
   for j in range(1, 26):
       if i != j and index < len(axes):
           ax = axes[index]
           for label in unique_labels:
               subset = df[df.iloc[:, 0] == label]
               ax.scatter(subset.iloc[:, i], subset.iloc[:, j], label=label_name_dict[label], color=color_dict[label])
           ax.set_title(f'Feature {i} vs Feature {j}')
           ax.legend()
           # 设置x轴和y轴的范围
           subset1 = df[df.iloc[:, 0] == unique_labels[0]]
           subset2 = df[df.iloc[:, 0] == unique_labels[1]]
           min_y = np.maximum(np.min(subset1.iloc[:, j]), np.min(subset2.iloc[:, j]))
           max_y = np.minimum(np.max(subset1.iloc[:, j]), np.max(subset2.iloc[:, j]))
           ax.set_xlim(9.84*1e7, 1e8)
           ax.set_ylim(9.6*1e7, 1.15e8)
           #ax.fill_between([np.min(df.iloc[:, i]), np.max(df.iloc[:, i])], min_y, max_y, color='grey', alpha=0.5)
           index += 1
# 调整子图之间的间距
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.4, hspace=0.4)
plt.tight_layout()

# 保存图像时设置DPI以降低图像大小
plt.savefig(save_path, dpi=100)
plt.close(fig)




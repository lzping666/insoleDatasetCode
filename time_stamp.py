import matplotlib.pyplot as plt
import pandas as pd

# 读取 CSV 文件并指定列名
column_names = ['timestamp', 'value1', 'value2', 'value3', 'value4', 'value5', 'value6', 'value7']
data = pd.read_csv('datasets/unlabeled/right_hwy_data.csv', sep=' ', names=column_names)

# 转换时间戳
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

# 创建一个简单的序列用于可视化
index = range(len(data))

# 绘制图形，仅显示时间戳
plt.figure(figsize=(10, 5))
plt.scatter(data['timestamp'], index, marker='o')

# 设置标题和标签
plt.title("Timestamps Visualization")
plt.xlabel("Time")
plt.ylabel("Index")

# 格式化 x 轴上的日期
plt.gcf().autofmt_xdate()

# 显示图形
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# 从CSV文件读取数据
data = pd.read_csv('../datasets/labeled/left_lys_data.csv')

# 提取压力值并转换为浮点数
pressures = [[] for _ in range(4)]  # 创建四个空列表来存储每个压力点的数据

for index in range(len(data)):
    # 获取当前行的字符串
    row_string = data.iloc[index, 0]  # 假设数据在第一列

    # 分割字符串并获取最后四个数字
    numbers = row_string.split()
    last_four_numbers = [float(x) for x in numbers[-4:]]

    # 将每个压力点的数据分别添加到对应的列表中
    for i in range(4):
        pressures[i].append(last_four_numbers[i])

# 创建图形
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# 绘制四个子图
for i in range(4):
    row = i // 2
    col = i % 2
    axs[row, col].plot(pressures[i], label=f'Pressure Point {i+1}')
    axs[row, col].set_title(f'Pressure Point {i+1}')
    axs[row, col].set_xlabel('Time')
    axs[row, col].set_ylabel('Pressure')
    axs[row, col].legend()

# 调整子图间距
plt.tight_layout()

# 显示图形
plt.show()
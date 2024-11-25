import math

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
from matplotlib.image import imread

# 从CSV文件读取数据
data = pd.read_csv('../datasets/labeled/left_lys_data.csv')

# 假设CSV文件中每行是一个字符串，数字之间用空格隔开
positions = [(20, 1.4), (16, 1.25), (14, 3.2), (3.6, 2.2)]

# 提取压力值并转换为浮点数
pressures = []
for index in range(len(data)):
    # 获取当前行的字符串
    row_string = data.iloc[index, 0]  # 假设数据在第一列

    # 分割字符串并获取最后四个数字
    numbers = row_string.split()
    last_four_numbers = [float(x) for x in numbers[-4:]]

    pressures.extend(last_four_numbers)

# 创建图形
fig, ax = plt.subplots(figsize=(6, 12))

# 读取背景图片
img = imread('datasets/img/2.jpg')

# 获取图片尺寸
img_height, img_width = img.shape[:2]
ax.set_xlim(0, img_width)
ax.set_ylim(0, img_height)

ax.set_xticks(range(11))
ax.set_yticks(range(11))
ax.grid(True, which='both', linestyle='--', linewidth=0.5)


# 调整图形
ax.set_aspect('equal')
ax.axis('off')
plt.title('insole pressure')


def update(frame):
    # 清除上一帧的图形
    ax.cla()
    # 绘制背景图片
    ax.imshow(img, extent=[0, img_width, 0, img_height], aspect='auto')  # 设置图片显示范围与坐标轴一致
    # 绘制网格背景 (因为上一行代码清除了所有图形，所以需要重新绘制)
    ax.set_xlim(0, img_width)
    ax.set_ylim(0, img_height)

    # 计算缩放比例
    x_scale = img_width / 10
    y_scale = img_height / 10

    ax.set_aspect('equal')

    ax.set_xticks(range(11))
    ax.set_yticks(range(11))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axis('off')
    plt.title('insole pressure')

    # 计算当前帧需要显示的数据索引
    start_index = frame * 4  # 每帧显示 4 个圆

    # 绘制四个圆形
    for i in range(4):
        # 检查索引是否超出数据范围
        if start_index + i < len(pressures):
            pressure = pressures[start_index + i]
            (x, y) = positions[i]
            radius = math.log2((pressure+1) * 1000)
            circle = plt.Circle((y * y_scale, x * x_scale),radius, color='red', alpha=0.5)
            ax.add_artist(circle)

    # 返回更新后的图形对象
    return ax,

# 创建动画对象
ani = animation.FuncAnimation(fig, update, frames=len(pressures) // 4, interval=50)

# 显示动画
plt.show()


import numpy as np
import pandas as pd
import random
import scipy.signal as signal

# def random_downsample(insole_data, original_freq=40, target_freq=30):
#     """随机剔除数据将鞋垫数据降采样到目标频率，并返回字符串列表。"""
#
#     downsampled_data_strings = []
#     for i in range(0, len(insole_data), original_freq):
#         second_data = insole_data[i:i + original_freq]
#         if len(second_data) < original_freq:
#             continue
#
#         indices_to_keep = random.sample(range(original_freq), target_freq)
#         indices_to_keep.sort()
#
#         second_data_downsampled = [second_data[j] for j in indices_to_keep] # 保持为字符串
#
#         downsampled_data_strings.extend(second_data_downsampled) # extend 字符串列表
#
#
#     return downsampled_data_strings # 返回字符串列表


def systematic_downsample(insole_data, original_freq=38, target_freq=30):
    """系统性降采样，保持时序均匀"""

    # 计算采样间隔
    step = original_freq / target_freq

    downsampled_data = []
    cumulative_index = 0

    for i in range(0, len(insole_data), original_freq):
        second_data = insole_data[i:i + original_freq]
        if len(second_data) < original_freq:
            continue

        # 计算这一秒需要的采样点
        indices = np.arange(0, original_freq, step)
        indices = np.floor(indices).astype(int)
        indices = indices[:target_freq]  # 确保只取target_freq个点

        second_data_downsampled = [second_data[j] for j in indices]
        downsampled_data.extend(second_data_downsampled)

    return downsampled_data

def main():
    """主函数，读取 CSV 文件并降采样鞋垫数据。"""

    try:
        # 从 CSV 文件读取鞋垫数据，并转换为字符串列表
        # insole_df = pd.read_csv("../datasets/labeled/right_data_fixed.csv", sep=",")
        insole_df = pd.read_csv("../datasets/labeled/right_data_fixed.csv", sep=",", usecols=range(8))  # 读取前8列
        insole_data = insole_df.values.tolist() # 先转换为列表

        insole_data_strings = []
        for row in insole_data:
            insole_data_strings.append(" ".join(map(str, row))) # 将每一行转换为字符串

        # 获取原始频率和目标频率 (可以从配置文件或用户输入获取)
        original_freq = 38
        target_freq = 30

        # 降采样鞋垫数据 (保持为字符串列表)
        # 降采样鞋垫数据 (现在 insole_data_strings 是字符串列表)
        # downsampled_data_strings = random_downsample(insole_data_strings, original_freq, target_freq)
        downsampled_data_strings = systematic_downsample(insole_data_strings, original_freq, target_freq)
        # 将字符串数据转换为数值类型
        downsampled_data_numeric = []
        for line in downsampled_data_strings:  # 直接处理字符串列表
            parts = line.split()
            try:
                # numeric_parts = [float(x) for x in parts[1:]]
                # downsampled_data_numeric.append(np.array([int(parts[0])] + numeric_parts))
                numeric_parts = [float(x) for x in parts]  # 将所有部分转换为浮点数
                downsampled_data_numeric.append(np.array(numeric_parts))  # 将每一行转换为 numpy 数组
            except ValueError as e:
                print(f"跳过无效行: {line}, 错误: {e}")
                continue

        downsampled_data = np.array(downsampled_data_numeric)  # 最后转换为 NumPy 数组

        # 保存降采样后的数据 (现在保存的是数值数据)
        # np.savetxt("../datasets/downsampled_insole_data2.csv", downsampled_data, delimiter=",", fmt='%d,%f,%f,%f,%f,%f,%f,%f') # 指定保存格式
        # 保存降采样后的数据 (现在保存的是数值数据)
        np.savetxt("../datasets/downsampled_insole_data5.csv", downsampled_data, delimiter=",",
                   fmt='%f,%f,%f,%f,%f,%f,%f,%f')
        print(f"原始数据形状: {insole_data.shape}")
        # print(f"降采样后数据形状: {downsampled_data.shape}")
        print("降采样完成，数据已保存到 downsampled_insole_data.csv")

    except FileNotFoundError:
        print(f"错误：文件 insole_data.csv 不存在。请检查文件路径。")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()
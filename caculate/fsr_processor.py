import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_fsr_data(csv_file_path):
    """
    读取FSR数据文件

    Parameters:
    csv_file_path: FSR数据文件的路径

    Returns:
    timestamps: 时间戳数组
    fsr_values: FSR压力值数组 shape: (frames, 4)
    """
    # 读取CSV文件，假设没有表头，用空格分隔
    # df = pd.read_csv(csv_file_path, header=None, delimiter=' ')
    df = pd.read_csv(csv_file_path, sep=",", header=None, dtype=np.float64)  # 明确指定 header=None
    # 提取时间戳（第一列）
    timestamps = df[0].values

    # 提取最后四列作为FSR值
    fsr_values = df.iloc[:, -4:].values

    return timestamps, fsr_values


def process_fsr_data(csv_file_path):
    """
    处理FSR数据并展示基本信息
    """
    # 读取数据
    timestamps, fsr_values = load_fsr_data(csv_file_path)

    print(f"数据总帧数: {len(timestamps)}")
    print("\nFSR数据形状:", fsr_values.shape)
    print("\n前5帧FSR数据:")
    print(fsr_values[:5])

    # 计算采样率
    time_diff = np.diff(timestamps)
    avg_sample_rate = 1000 / np.mean(time_diff)  # 转换为Hz
    print(f"\n平均采样率: {avg_sample_rate:.2f} Hz")

    return timestamps, fsr_values


def preprocess_fsr_data(fsr_values, smooth_window=5):
    """
    预处理FSR数据

    Parameters:
    fsr_values: 原始FSR数据
    smooth_window: 平滑窗口大小

    Returns:
    processed_values: 归一化后的FSR数据
    scaler: 归一化的MinMaxScaler对象
    """
    # 移动平均平滑
    smoothed_values = pd.DataFrame(fsr_values).rolling(
        window=smooth_window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values

    # 初始化 MinMaxScaler
    scaler = MinMaxScaler()

    # 使用原始数据进行 fit
    scaler.fit(fsr_values)  # 这里确保 scaler 的范围是基于原始数据的范围

    # 对平滑后的数据进行归一化
    processed_values = scaler.transform(smoothed_values)

    return processed_values, scaler


# def preprocess_fsr_data(fsr_values, smooth_window=5):
#     """
#     预处理FSR数据
#
#     Parameters:
#     fsr_values: 原始FSR数据
#     smooth_window: 平滑窗口大小
#
#     Returns:
#     processed_values: 处理后的FSR数据
#     """
#     # 移动平均平滑
#     processed_values = pd.DataFrame(fsr_values).rolling(
#         window=smooth_window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
#
#     # 归一化
#     processed_values = (processed_values - processed_values.min(axis=0)) / \
#                        (processed_values.max(axis=0) - processed_values.min(axis=0))
#
#     return processed_values

# def preprocess_fsr_data(fsr_values, smooth_window=5):
#     """
#     预处理FSR数据
#
#     Parameters:
#     fsr_values: 原始FSR数据
#     smooth_window: 平滑窗口大小
#
#     Returns:
#     processed_values: 处理后的FSR数据
#     data_min: 每个传感器的最小值
#     data_max: 每个传感器的最大值
#     """
#     # 移动平均平滑
#     processed_values = pd.DataFrame(fsr_values).rolling(
#         window=smooth_window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
#
#     # 保存原始范围
#     data_min = processed_values.min(axis=0)
#     data_max = processed_values.max(axis=0)
#
#     # 归一化
#     processed_values = (processed_values - data_min) / (data_max - data_min)
#
#     return processed_values, data_min, data_max

# 使用示例
if __name__ == "__main__":
    # 替换为您的CSV文件路径
    csv_file_path = "../datasets/labeled/right_lys_data.csv"
    timestamps, fsr_values = process_fsr_data(csv_file_path)
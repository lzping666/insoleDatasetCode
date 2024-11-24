import numpy as np
import pandas as pd


def load_pose_data(csv_file_path):
    """
    读取姿态数据
    """
    # 读取CSV文件，跳过第一行（列名）
    df = pd.read_csv(csv_file_path, low_memory=False)

    # 只选择包含坐标的列（x, y, z），排除visibility列和其他列
    coord_columns = [col for col in df.columns if any(x in col for x in ['_x', '_y', '_z'])
                     and 'visibility' not in col]

    # 删除第一行（如果它包含列名）
    if df.iloc[0].equals(pd.Series(df.columns)):
        df = df.iloc[1:]

    # 将数据转换为浮点数
    pose_data = df[coord_columns].astype(float).values

    # 重塑数据为 (samples, 33, 3) 的形状
    num_samples = pose_data.shape[0]
    pose_data = pose_data.reshape(num_samples, -1, 3)

    return pose_data

#对姿态数据进行预处理，包括平滑处理和归一化处理。
def preprocess_pose_data(pose_data, smooth_window=5):
    """
    预处理姿态数据
    Parameters:
    pose_data: 原始姿态数据，shape: (frames, 33, 3)
    smooth_window: 平滑窗口大小

    Returns:
    processed_pose: 处理后的姿态数据，shape: (frames, 33, 3)
    """
    frames, num_keypoints, num_coordinates = pose_data.shape

    # 直接使用原始数据，不需要reshape
    pose_values = pose_data

    # 对每个维度应用移动平均平滑
    processed_pose = np.zeros_like(pose_values)
    for i in range(num_keypoints):
        for j in range(num_coordinates):
            values = pd.Series(pose_values[:, i, j])
            smoothed = values.rolling(window=smooth_window, center=True).mean()
            processed_pose[:, i, j] = smoothed.fillna(method='bfill').fillna(method='ffill')

    # 对每个关键点的坐标进行归一化
    for i in range(num_keypoints):
        for j in range(num_coordinates):
            min_val = processed_pose[:, i, j].min()
            max_val = processed_pose[:, i, j].max()
            if max_val > min_val:  # 避免除以零
                processed_pose[:, i, j] = (processed_pose[:, i, j] - min_val) / (max_val - min_val)

    return processed_pose
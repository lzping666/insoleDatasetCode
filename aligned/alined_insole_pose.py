import pandas as pd
import numpy as np


def downsample_insole_to_pose(pose_file, insole_file, output_file):
    """根据姿态数据的时间戳对鞋垫数据进行降采样。"""
    try:
        pose_df = pd.read_csv(pose_file)
        insole_df = pd.read_csv(insole_file)

        pose_df['timestamp'] = pd.to_datetime(pose_df['timestamp'].astype(float), unit='s')
        insole_df['timestamp'] = pd.to_datetime(insole_df['timestamp'].astype(float), unit='s')
    except (FileNotFoundError, ValueError) as e:
        print(f"Error reading or processing data: {e}")
        return

    # 即使数据已经对齐，仍然需要排序以确保正确查找最近邻
    insole_df = insole_df.sort_values('timestamp').reset_index(drop=True)

    pose_timestamps = pose_df['timestamp'].values.astype(np.int64)
    insole_timestamps = insole_df['timestamp'].values.astype(np.int64)

    nearest_indices = []
    for pose_ts in pose_timestamps:
        time_diffs = np.abs(insole_timestamps - pose_ts)
        min_idx = np.argmin(time_diffs)
        nearest_indices.append(min_idx)

    # 检查并警告潜在问题
    if len(set(nearest_indices)) < 10:  # 检查唯一索引的数量是否过少
        print("警告：匹配到的鞋垫数据行数过少，请检查数据和时间戳！")


    downsampled_insole_df = insole_df.iloc[nearest_indices].reset_index(drop=True)

    # 将降采样后的鞋垫数据与姿态数据合并
    aligned_df = pd.concat([pose_df, downsampled_insole_df.drop(columns=['timestamp'])], axis=1)

    # 保存结果
    aligned_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"降采样完成，已保存到: {output_file}")

if __name__ == '__main__':
    username = "zsb"
    pose_file = f"../datasets/{username}/pose1.csv"  # 姿态数据 CSV 文件路径
    insole_file = f"../datasets/{username}/right_data_fixed.csv"  # 鞋垫数据 CSV 文件路径
    output_file = f"../datasets/{username}/aligned_insole_data.csv"  # 输出对齐后的鞋垫数据 CSV 文件路径

    downsample_insole_to_pose(pose_file, insole_file, output_file)
import pandas as pd
import numpy as np

def merge_data(pose1,pose2, left_data, right_data,out_file,have_com_data,start_time,com8_data=None,com11_data=None,com18_data=None):
    # 读取 CSV 文件
    pose1_df = pd.read_csv(pose1)
    pose2_df = pd.read_csv(pose2)

    pose1_df_filtered = pose1_df.drop(columns=['video1_fid'], errors='ignore')
    pose2_df_filtered = pose2_df.drop(columns=['video2_fid'], errors='ignore')

    left_data_df = pd.read_csv(left_data)
    right_data_df = pd.read_csv(right_data)

    # 合并数据框，使用 timestamp 作为关键列
    merged_df = pd.merge(pose1_df_filtered, left_data_df, on='timestamp', how='outer')
    merged_df = pd.merge(merged_df,pose2_df_filtered,on='timestamp',how='outer')
    merged_df = pd.merge(merged_df, right_data_df, on='timestamp', how='outer')

    #如果拥有传感器数据的话,就把传感器数据也合并
    if have_com_data:
        com8_df = pd.read_csv(com8_data)
        com11_df = pd.read_csv(com11_data)
        com18_df = pd.read_csv(com18_data)
        merged_df = pd.merge(merged_df, com8_df, on='timestamp', how='outer')
        merged_df = pd.merge(merged_df, com11_df, on='timestamp', how='outer')
        merged_df = pd.merge(merged_df, com18_df, on='timestamp', how='outer')

    # 只保留 timestamp 大于 start_time 的行
    merged_df = merged_df[merged_df['timestamp'] > start_time]

    # 按照 timestamp 排序
    merged_df.sort_values(by='timestamp', inplace=True)

    # 重置索引
    merged_df.reset_index(drop=True, inplace=True)

    # 保存到 final.csv
    merged_df.to_csv(out_file, index=False)



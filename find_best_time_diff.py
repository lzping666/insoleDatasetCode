import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import fix_foot_data
from visualize import visualization


def find_best_time_diff(pose_file, left_file, start_time, end_time):
    # 读取数据
    pose_df = pd.read_csv(pose_file)
    left_df = pd.read_csv(left_file)

    #转换时间戳为 datetime 格式  加上pd.to_numeric防止出warning
    pose_df['timestamp'] = pd.to_datetime(pd.to_numeric(pose_df['timestamp']), unit='s')
    left_df['timestamp'] = pd.to_datetime(pd.to_numeric(left_df['timestamp']), unit='s')

    # 提取 y27 和压力数据
    left_df['total_pressure'] = left_df[['left_f1', 'left_f2', 'left_f3', 'left_f4']].sum(axis=1)

    #定义时间范围
    start_time = pd.to_datetime(pd.to_numeric(start_time), unit='s')
    end_time = pd.to_datetime(pd.to_numeric(end_time), unit='s')

    # 过滤 pose_df 的时间范围
    pose_df = pose_df[(pose_df['timestamp'] >= start_time) & (pose_df['timestamp'] <= end_time)]

    # 计算可能的时间差
    time_diffs = np.arange(-80, 80, 0.001)  # 从 -40 到 40 秒，步长为 0.001 秒
    best_corr = -1
    best_time_diff = None

    timesdif=[]
    corrs=[]
    print("正在匹配鞋垫数据")
    for time_diff in time_diffs:
        shifted_left_df = left_df.copy()
        shifted_left_df['timestamp'] += pd.to_timedelta(time_diff, unit='s')

        # 过滤 shifted_left_df 的时间范围
        shifted_left_df = shifted_left_df[(shifted_left_df['timestamp'] >= start_time) &
                                           (shifted_left_df['timestamp'] <= end_time)]

        # 合并数据以对齐时间戳
        combined_df = pd.merge_asof(pose_df.sort_values('timestamp'),
                                     shifted_left_df.sort_values('timestamp'),
                                     on='timestamp', direction='nearest')

        # 计算相关系数
        pose_values = (combined_df['video2_y27']-combined_df['video2_y11']).dropna()
        pressure_values = combined_df['total_pressure'].dropna()

        if len(pose_values) > 1 and len(pressure_values) > 1:
            corr, _ = pearsonr(pose_values, pressure_values)
            corrs.append(corr)
            timesdif.append(time_diff)
            if corr > best_corr:
                best_corr = corr
                best_time_diff = time_diff
    plt.plot(timesdif, corrs)
    plt.show()
    return best_time_diff, best_corr

if __name__ == '__main__':
    user_name = 'cwh'
    start_time = float(1726817528)
    end_time= float(1726817563)
    best_time_diff, best_corr = find_best_time_diff(
        f'{user_name}/pose2.csv',
        f'{user_name}/left_data.csv',
        start_time,
        end_time
    )
    print(best_time_diff)
    fix_foot_data.footdata2csv(f'{user_name}/right_data.csv',
                               f'{user_name}/right_data_fixed.csv',
                               best_time_diff,
                               'right')

    fix_foot_data.footdata2csv(f'{user_name}/left_data.csv',
                               f'{user_name}/left_data_fixed.csv',
                               best_time_diff,
                               'left')

    visualization.visualization(f'{user_name}/left_data_fixed.csv',
                                f'{user_name}/pose2.csv',
                                start_time,
                                end_time)
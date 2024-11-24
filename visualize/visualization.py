import pandas as pd
import matplotlib.pyplot as plt

#可视化两个数据是否相对匹配

#由于压力数据最大值会到达几百而y的坐标都是0.几，所以需要归一化
def normalize(data):
    return (data - data.min()) / (data.max() - data.min())

def visualization(foot_file, video_file,start_time, end_time):
    foot_data = pd.read_csv(foot_file)
    video_data = pd.read_csv(video_file)

    foot_data = foot_data[(foot_data['timestamp'] >= start_time) & (foot_data['timestamp'] <= end_time)]
    video_data = video_data[(video_data['timestamp'] >= start_time) & (video_data['timestamp'] <= end_time)]

    y27=video_data['video2_y27']
    time = video_data['timestamp']
    plt.plot(time,normalize(y27))
    plt.plot(foot_data['timestamp'],normalize(foot_data['left_total_pressure']),color='red')
    plt.show()

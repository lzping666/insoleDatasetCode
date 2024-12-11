import pandas as pd

import pandas as pd

def remove_pose_data(input_file, output_file):
    """
    从对齐后的文件中清除姿态数据，只保留时间戳和鞋垫数据。

    Parameters:
    input_file (str): 输入对齐后的文件路径
    output_file (str): 输出仅包含时间戳和鞋垫数据的文件路径
    """
    # 读取对齐后的文件
    aligned_df = pd.read_csv(input_file)

    # 明确鞋垫数据的列名
    insole_columns = [
        'right_ax', 'right_ay', 'right_az',
        'right_f1', 'right_f2', 'right_f3', 'right_f4'
    ]

    # 筛选时间戳和鞋垫数据列
    filtered_df = aligned_df[['timestamp'] + insole_columns]

    # 保存到新的文件
    filtered_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"新文件已生成，仅包含时间戳和鞋垫数据，保存到: {output_file}")


if __name__ == '__main__':
    username = 'zsb'
    input_file = f"../datasets/{username}/aligned_insole_data.csv"  # 输入对齐后的文件路径
    output_file = f"../datasets/{username}/filtered_insole_data1.csv"  # 输出仅包含时间戳和鞋垫数据的文件路径

    remove_pose_data(input_file, output_file)
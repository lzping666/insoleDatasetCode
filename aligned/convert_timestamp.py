import pandas as pd

def convert_timestamp_format(input_file, output_file):
    """
    将文件中的时间戳从 datetime 格式转换为 UNIX 时间戳格式，并设置为 float 类型。

    Parameters:
    input_file (str): 输入文件路径，包含 datetime 格式的时间戳
    output_file (str): 输出文件路径，包含 UNIX 时间戳格式的时间戳（float 类型）
    """
    # 读取文件
    df = pd.read_csv(input_file)

    # 转换时间戳为 UNIX 时间戳（秒级，带小数点）并强制转换为 float 类型
    df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') / 1e9
    df['timestamp'] = df['timestamp'].astype(float)

    # 保存转换后的文件
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"时间戳已转换为 float 类型，结果保存到: {output_file}")

# 示例调用
if __name__ == '__main__':
    input_file = "../datasets/filtered_insole_data3.csv"  # 输入文件路径
    output_file = "../datasets/filtered_insole_data4.csv"  # 输出文件路径

    convert_timestamp_format(input_file, output_file)
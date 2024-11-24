import pandas as pd


def footdata2csv(input_file, output_file,foot):
    # 读取数据，假设数据都在第一列
    data = pd.read_csv(input_file, header=None)

    # 按空格分割数据
    split_data = data[0].str.split(expand=True)

    split_data[0] = split_data[0].str.slice(0, -3) + '.' + split_data[0].str.slice(-3)
    # 添加列名
    split_data.columns = ['timestamp', foot+'_ax', foot+'_ay', foot+'_az', foot+'_f1', foot+'_f2', foot+'_f3', foot+'_f4']

    # 保存为新的CSV文件
    split_data.to_csv(output_file, index=False)

def comdata2csv(input_file, output_file,com):
    # 读取数据，假设数据都在第一列
    data = pd.read_csv(input_file, header=0)
    # 添加列名
    data.columns = ['timestamp', com + '_ax', com + '_ay', com + '_az', com + '_wx', com + '_wy',
                          com + '_wz', com + '_hx', com+'_hy', com+'_hz']
    # 保存为新的CSV文件
    data.to_csv(output_file, index=False)

if __name__ == '__main__':
    comdata2csv('./zsb/com8_2024_09_30_15_15_40.csv','./zsb/com8.csv','com8')
import pandas as pd


# 对foot_data.csv文件加入totol_pressure列同时修正他的时间差
def footdata2csv(input_file, output_file,delta_time,foot):
    data = pd.read_csv(input_file)

    data[foot+'_total_pressure'] = data[[foot+'_f1', foot+'_f2', foot+'_f3', foot+'_f4']].sum(axis=1)

    data['timestamp']+=delta_time
    # 保存为新的CSV文件
    data.to_csv(output_file, index=False)


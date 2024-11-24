import getpose
from visualize import visualization
import fix_foot_data
import find_best_time_diff
import footdata2csv
import merge_data

if __name__ == '__main__':
    user_name = 'lkw'

    # 这里需要修改
    start_time, end_time = getpose.process_video(f'{user_name}/c2_2024_09_26_16_04_29.avi',
                                                 f'{user_name}/pose2.csv',
                                                 f'{user_name}/c2_2024_09_26_16_04_29.csv',True)
    getpose.process_video(f'{user_name}/c1_2024_09_26_16_03_45.avi',f'{user_name}/pose1.csv',
                                                 f'{user_name}/c1_2024_09_26_16_03_45.csv',False)

    footdata2csv.footdata2csv(f'{user_name}/left_{user_name}_data.csv',
                              f'{user_name}/left_data.csv',
                              'left')

    footdata2csv.footdata2csv(f'{user_name}/right_{user_name}_data.csv',
                              f'{user_name}/right_data.csv',
                              'right')

    best_time_diff, best_corr = find_best_time_diff.find_best_time_diff(
        f'{user_name}/pose2.csv',
        f'{user_name}/left_data.csv',
        start_time,
        end_time
    )

    print(f"Best time difference: {best_time_diff} ")

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

    footdata2csv.comdata2csv(f'{user_name}/com8_2024_09_26_16_03_49.csv',f'{user_name}/com8.csv','com8')
    footdata2csv.comdata2csv(f'{user_name}/com11_2024_09_26_16_03_48.csv',f'{user_name}/com11.csv','com11')
    footdata2csv.comdata2csv(f'{user_name}/com18_2024_09_26_16_03_50.csv',f'{user_name}/com18.csv','com18')

    merge_data.merge_data(f'{user_name}/pose1.csv',f'{user_name}/pose2.csv',
                          f'{user_name}/left_data_fixed.csv',
                          f'{user_name}/right_data_fixed.csv',
                          f'{user_name}/final.csv', True,
                          start_time,f'{user_name}/com8.csv',
                          f'{user_name}/com11.csv',
                          f'{user_name}/com18.csv')

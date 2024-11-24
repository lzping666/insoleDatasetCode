'''
@author:YULIANG
'''
import cv2
import csv
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
# 从指定文件中读取csv文件
def read_csv(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def calculate_angle(p1, p2, p3):
    """计算三点之间的夹角"""
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
    return angle


def is_t_pose_with_bent_arms(landmarks):
    """检查双臂弯曲90度且小臂高于大臂的姿势"""
    if landmarks is None:
        return False

    # 提取关键点
    shoulder_r = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    elbow_r = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    wrist_r = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    shoulder_l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    elbow_l = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    wrist_l = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

    # 计算角度
    angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
    angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)

    # 定义角度阈值和y坐标阈值
    angle_threshold = 30  # 90度角的容差范围
    # 检查手臂是否在90度左右弯曲，并且小臂的y坐标大于大臂的y坐标
    return (abs(angle_r - 90) < angle_threshold and
            abs(angle_l - 90) < angle_threshold and
            wrist_r.y < elbow_r.y and
            wrist_l.y < elbow_l.y)

# 写进csv数据
def write_csv(csv_path, data: list):
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def process_video(video_path, output_file,source_csv,is_video_2):
    cap = cv2.VideoCapture(video_path)
    start_time = end_time = -1
    if is_video_2:
        prefix="video2_"
    else:
        prefix="video1_"
    if is_video_2:
        pre_time= stats_time = -1
        #pre_time代表检测到三叉戟动作的时间戳,stats_time代表结束三叉戟动作的时间戳
        #我们需要保证短时间的三叉戟动作不会被识别(必须得坚持5s以上),且短时间的不做三叉戟动作也不会破坏正在做三叉戟的状态(至少得坚持2s以上)
        is_start = False
    mp_holistic = mp.solutions.holistic
    header = ['timestamp', prefix+'fid']
    for i in range(33):
        header.extend([f'{prefix}x{i}', f'{prefix}y{i}', f'{prefix}z{i}', f'{prefix}visibility{i}'])
    write_csv(output_file, header)
    csv_data = read_csv(source_csv)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for i in csv_data[1:]:
            ret, image = cap.read()
            if not ret:
                break
            timestamp , f_id = i[0] , i[1]

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Flatten the landmark data into separate columns
                if is_video_2 and (start_time == -1 or end_time == -1):
                    if is_t_pose_with_bent_arms(landmarks):
                        cv2.putText(image, "Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        if not is_start and (stats_time==-1 or float(timestamp)-float(stats_time)>2):
                            pre_time = timestamp
                        is_start = True
                    else:
                        if is_start:
                            stats_time=timestamp
                        if pre_time!=-1 and float(timestamp)-float(stats_time)>2 and float(stats_time)-float(pre_time)>5:
                            if start_time==-1:
                                start_time = stats_time
                                pre_time=-1
                            elif start_time!=-1 and end_time==-1:
                                end_time = stats_time
                                print("正在开始寻找踏步动作")
                        cv2.putText(image, "not", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        is_start = False
                data = [timestamp, f_id]
                for lm in landmarks:
                    data.extend([lm.x, lm.y, lm.z, lm.visibility])
                write_csv(output_file, data)


            cv2.imshow('Video Processing', image)
            if cv2.waitKey(10) == 27:
                break


    cap.release()
    cv2.destroyAllWindows()

    print(start_time)
    print(end_time)
    return float(start_time),float(end_time)



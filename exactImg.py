import cv2

# 打开视频文件
cap = cv2.VideoCapture('./video/lys.mp4')

# 循环读取视频帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 在这里可以对每一帧进行处理，例如保存为图片
    cv2.imwrite('./frame/frame_{}.jpg'.format(cap.get(cv2.CAP_PROP_POS_FRAMES)), frame)

    # 显示当前帧（可选）
    cv2.imshow('Frame', frame)

    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获对象并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
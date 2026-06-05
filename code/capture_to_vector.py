# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import json
import os
from datetime import datetime

# 确保文件夹存在
os.makedirs("screenshots", exist_ok=True)
os.makedirs("data", exist_ok=True)

# 初始化 MediaPipe Pose 模块
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# 用于存储所有帧的动作数据
motion_data = []

# 打开摄像头 (参考组员一的代码)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("摄像头打不开，请检查设备。")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("动作捕获已启动！")
print("全身入镜后开始做动作。")
print("按 's' 键保存带骨架的截图，按 'q' 键退出并生成 JSON 数据。")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("读取摄像头画面失败。")
        break

    # 镜像画面
    frame = cv2.flip(frame, 1)

    # 1. 将 BGR 转换为 RGB，因为 MediaPipe 需要 RGB 输入
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 2. 进行姿态检测
    results = pose.process(rgb_frame)

    # 3. 如果检测到人体关键点，绘制骨架并保存数据
    if results.pose_landmarks:
        # 在画面上绘制骨架
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # 提取当前帧的关键点数据
        frame_keypoints = []
        for landmark in results.pose_landmarks.landmark:
            frame_keypoints.append({
                "x": landmark.x,
                "y": landmark.y,
                "visibility": landmark.visibility
            })

        # 将当前帧数据存入总列表
        motion_data.append({
            "frame": frame_count,
            "keypoints": frame_keypoints
        })
        frame_count += 1

    # 在画面上加说明文字
    cv2.putText(frame, f"Frames captured: {frame_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Press 's' to screenshot, 'q' to save & quit", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Pose Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    # 按 's' 保存截图
    if key == ord('s'):
        screenshot_name = f"screenshots/pose_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(screenshot_name, frame)
        print(f"骨架截图已保存：{screenshot_name}")

    # 按 'q' 退出循环
    if key == ord('q'):
        print("捕获结束，正在保存数据...")
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

# 4. 将所有收集到的动作数据保存为 JSON 文件
json_path = "../data/motion_data.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(motion_data, f, indent=4)

print(f"动作数据已成功保存至：{json_path}")
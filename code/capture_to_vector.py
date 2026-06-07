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

# 打开摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("摄像头打不开，请检查设备。")
    exit()

# 统一分辨率：640x480
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# ================= 新增：视频保存配置 =================
# 设置保存路径和编码格式 (mp4v 是广泛支持的 MP4 编码)
# 改用对网页浏览器支持最完美的 WebM 格式 (VP8 编码)
# 改回 mp4 后缀
video_path = os.path.join("data", "source_video.mp4")
# 使用 avc1 (即 H.264 编码)，这是所有现代浏览器原生支持的标准
fourcc = cv2.VideoWriter_fourcc(*'avc1')
# 设定保存帧率为 30 FPS (与网页端的 ESTIMATED_FPS = 30 保持完全一致)
out = cv2.VideoWriter(video_path, fourcc, 30.0, (FRAME_WIDTH, FRAME_HEIGHT))
# ===================================================

print("动作捕获已启动！")
print("全身入镜后开始做动作。")
print("按 's' 键保存带骨架的截图，按 'q' 键退出并生成 JSON 和 MP4 数据。")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("读取摄像头画面失败。")
        break

    # 1. 镜像画面，让用户看着自然
    frame = cv2.flip(frame, 1)

    # ================= 视频录制核心修复 =================
    # 【新增】：强行将画面缩放到 640x480！
    # 防止摄像头不听话输出 720p 导致 OpenCV 生成损坏的空文件
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # 写入文件
    out.write(frame)
    # ===================================================

    # 2. 转换颜色空间供 MediaPipe 处理
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_frame)

    # 3. 提取并绘制骨架
    if results.pose_landmarks:
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

    # 在画面上加说明文字（注意：这些字只会显示在预览窗口，不会录进 mp4 里）
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
        print(f"骨架截图已保存: {screenshot_name}")

    # 按 'q' 退出
    elif key == ord('q'):
        print("捕获结束，正在保存数据...")
        break

# === 退出清理工作 ===
# 将关键点数据写入 JSON
json_path = os.path.join("data", "motion_data.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(motion_data, f, indent=4)

print(f"动作坐标数据已保存至: {json_path}")
print(f"原始视频文件已保存至: {video_path}")

# 释放资源
cap.release()
out.release() # 释放视频写入对象
cv2.destroyAllWindows()
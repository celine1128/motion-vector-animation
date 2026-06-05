import cv2
import os
from datetime import datetime

# 创建截图保存文件夹
os.makedirs("screenshots", exist_ok=True)

# Windows 上加 CAP_DSHOW 可以减少摄像头打开慢、黑屏的问题
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("默认摄像头 0 打不开，尝试摄像头 1。")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("摄像头仍然打不开。请检查：")
    print("1. 摄像头是否被 QQ、微信、腾讯会议占用")
    print("2. Windows 是否允许应用访问摄像头")
    print("3. 电脑是否有外接摄像头")
    exit()

# 设置画面大小
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("摄像头打开成功。")
print("按 s 保存截图，按 q 退出。")

while True:
    ret, frame = cap.read()

    if not ret:
        print("读取摄像头画面失败。")
        break

    # 镜像画面，自己看起来更自然
    frame = cv2.flip(frame, 1)

    # 在画面上加说明文字
    cv2.putText(
        frame,
        "Camera Test - Press s to save, q to quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        filename = f"screenshots/camera_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, frame)
        print(f"截图已保存：{filename}")

    if key == ord('q'):
        print("退出摄像头测试。")
        break

cap.release()
cv2.destroyAllWindows()

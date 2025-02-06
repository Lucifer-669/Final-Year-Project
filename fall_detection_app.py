# app.py
import cv2
import time
import numpy as np
import gradio as gr
from typing import Iterator
from ultralytics import YOLO

# ========================= 系统配置 =========================
MODEL_PATH = "runs/detect/fall_detection_optimized/weights/best.pt"
FALL_CLASS_ID = 0  # 数据集中"摔倒"类别的ID
CONF_THRESHOLD = 0.8  # 检测置信度阈值
TRIGGER_FRAMES = 15  # 连续触发帧数（防误报）
DEFAULT_CAMERA = 0  # 默认摄像头设备号


# ===========================================================

class FallDetectionSystem:
    def __init__(self):
        # 加载训练好的YOLOv8模型
        self.model = YOLO(MODEL_PATH)
        self.fall_counter = 0
        self.alarm_status = False
        self.alarm_start_time = 0
        self.alarm_duration = 10  # 警报持续时间（秒）

    def process_frame(self, frame: np.ndarray) -> tuple:
        """处理单帧图像"""
        results = self.model(frame, verbose=False)[0]
        fall_detected = False
        annotated_frame = results.plot()

        # 检测逻辑
        for box in results.boxes:
            if int(box.cls) == FALL_CLASS_ID and float(box.conf) > CONF_THRESHOLD:
                self.fall_counter += 1
                if self.fall_counter >= TRIGGER_FRAMES:
                    fall_detected = True
                    self.alarm_status = True
                    self.alarm_start_time = time.time()
                    self.fall_counter = 0
                break
        else:
            self.fall_counter = max(0, self.fall_counter - 1)

        if self.alarm_status and (time.time() - self.alarm_start_time) > self.alarm_duration:
            self.alarm_status = False

        return self.alarm_status, annotated_frame


# 初始化检测系统
detector = FallDetectionSystem()


# ====================== Gradio界面 ======================
def generate_alert_html(alarm_status: bool) -> str:
    """生成警报提示的HTML代码"""
    if alarm_status:
        return """
        <div style='
            color: white;
            background: #ff4444;
            font-size: 24px;
            border: 3px solid darkred;
            padding: 15px;
            border-radius: 8px;
            animation: blink 1s infinite;
        '>
        🚨 检测到摔倒事件！
        </div>
        <style>
            @keyframes blink {
                0% {opacity: 1;}
                50% {opacity: 0.5;}
                100% {opacity: 1;}
            }
        </style>
        """
    return ""


def image_inference(image: np.ndarray) -> tuple:
    """图片检测处理"""
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    alarm_status, annotated_frame = detector.process_frame(frame)
    return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), generate_alert_html(alarm_status)


def video_inference(video_path: str) -> Iterator[tuple]:
    """视频检测处理"""
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        alarm_status, annotated_frame = detector.process_frame(frame)
        yield cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), generate_alert_html(alarm_status)
    cap.release()


def webcam_inference(camera_device: int) -> Iterator[tuple]:
    """USB摄像头实时检测"""
    cap = cv2.VideoCapture(camera_device)
    if not cap.isOpened():
        yield None, "<div style='color: red'>错误：无法打开摄像头设备，请检查连接！</div>"
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        alarm_status, annotated_frame = detector.process_frame(frame)
        yield cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), generate_alert_html(alarm_status)
    cap.release()


# 构建交互界面
with gr.Blocks(theme=gr.themes.Soft(), title="摔倒检测系统") as app:
    gr.Markdown("# 🚨 智能摔倒检测系统（完整版）")
    gr.Markdown("支持图像/视频文件检测和USB摄像头实时检测")

    # 全局摄像头设置
    with gr.Row():
        camera_device = gr.Number(
            label="摄像头设备号",
            value=DEFAULT_CAMERA,
            precision=0,
            minimum=0,
            maximum=10,
            step=1,
            interactive=True
        )
        gr.HTML("""
        <div style="color: #666; margin-top: 8px">
        设备号说明：
        <ul>
            <li>0 ➔ 默认摄像头（笔记本电脑内置）</li>
            <li>1 ➔ 第一个外接USB摄像头</li>
            <li>2 ➔ 第二个外接摄像头（如有）</li>
        </ul>
        </div>
        """)

    # 标签页布局
    with gr.Tabs():
        # 图片检测标签
        with gr.Tab("📷 图片检测"):
            with gr.Row():
                img_input = gr.Image(label="上传图片", type="numpy")
                img_output = gr.Image(label="检测结果", interactive=False)
            alert_output_img = gr.HTML()
            img_btn = gr.Button("开始检测", variant="primary")
            img_btn.click(image_inference, [img_input], [img_output, alert_output_img])

        # 视频检测标签
        with gr.Tab("🎥 视频检测"):
            with gr.Row():
                vid_input = gr.Video(label="上传视频", sources=["upload"])
                vid_output = gr.Image(label="检测结果", streaming=True)
            alert_output_vid = gr.HTML()
            vid_btn = gr.Button("开始检测", variant="primary")
            vid_btn.click(video_inference, [vid_input], [vid_output, alert_output_vid])

        # 实时摄像头标签
        with gr.Tab("📹 USB摄像头"):
            with gr.Row():
                webcam_output = gr.Image(label="实时画面", streaming=True)
                alert_output_webcam = gr.HTML()
            webcam_btn = gr.Button("启动摄像头", variant="primary")
            webcam_btn.click(
                fn=webcam_inference,
                inputs=[camera_device],
                outputs=[webcam_output, alert_output_webcam],
                show_progress="hidden"
            )

# 启动应用
if __name__ == "__main__":
    app.launch(
        server_port=7860,
        show_error=True,
        favicon_path="./icon.ico"
    )
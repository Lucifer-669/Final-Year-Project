import cv2
import time
import numpy as np
import gradio as gr
from typing import Iterator
from ultralytics import YOLO
import pygame

# 初始化 pygame.mixer
pygame.mixer.init()

# ========================= 系统配置 =========================
MODEL_PATH = "runs/detect/fall_detection_optimized/weights/best.pt"
FALL_CLASS_ID = 0
INIT_CONF_THRESHOLD = 0.5
TRIGGER_FRAMES = 15
DEFAULT_CAMERA = 0
ALARM_SOUND_PATH = r"C:\Users\AC_Chris\Desktop\alarm.wav"
HISTORY_FILE_PATH = r"C:\Users\AC_Chris\Desktop\fall_history.txt"

# ===========================================================
class FallDetectionSystem:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.fall_counter = 0
        self.alarm_status = False
        self.alarm_start_time = 0
        self.alarm_duration = 10
        self.conf_threshold = INIT_CONF_THRESHOLD
        self.history = []
        self.save_history = True

    def set_conf_threshold(self, threshold: float):
        self.conf_threshold = threshold

    def toggle_auto_save(self, save: bool):
        self.save_history = save

    def clear_history(self):
        self.history = []
        return self.history

    def save_to_file(self):
        try:
            with open(HISTORY_FILE_PATH, "w", encoding="utf-8") as f:
                for event in self.history:
                    f.write(f"{event['time']} | 置信度: {event['confidence']:.2f} | 状态: {event['status']}\n")
            return True
        except Exception as e:
            print(f"保存失败: {str(e)}")
            return False

    def process_frame(self, frame: np.ndarray) -> tuple:
        results = self.model(frame, verbose=False)[0]
        fall_detected = False
        annotated_frame = results.plot()
        for box in results.boxes:
            if int(box.cls) == FALL_CLASS_ID and float(box.conf) > self.conf_threshold:
                self.fall_counter += 1
                if self.fall_counter >= TRIGGER_FRAMES:
                    fall_detected = True
                    self.alarm_status = True
                    self.alarm_start_time = time.time()
                    self.fall_counter = 0
                    if self.save_history:
                        # 判断是 "falling" 还是 "fallen"
                        if self.fall_counter == TRIGGER_FRAMES:
                            status = "falling"
                        else:
                            status = "fallen"
                        self.history.append({
                            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                            "confidence": float(box.conf),
                            "status": status
                        })
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.stop()
                    pygame.mixer.music.load(ALARM_SOUND_PATH)
                    pygame.mixer.music.play()
                break
        else:
            self.fall_counter = max(0, self.fall_counter - 1)
        if self.alarm_status and (time.time() - self.alarm_start_time) > self.alarm_duration:
            self.alarm_status = False
            pygame.mixer.music.stop()
        return self.alarm_status, annotated_frame


detector = FallDetectionSystem()

# ====================== Gradio界面 ======================
def generate_alert_html(alarm_status: bool) -> str:
    if alarm_status:
        return """
        <div style='
            color: white;
            background: linear-gradient(45deg, #ff5555, #ff8888);
            font-size: 24px;
            border: 3px solid darkred;
            padding: 15px;
            border-radius: 8px;
            animation: blink 1s infinite;
            box-shadow: 0 0 10px rgba(255, 0, 0, 0.5);
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

def update_history_html(history):
    rows = ""
    for event in history:
        rows += f"""
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #ccc;">{event['time']}</td>
            <td style="padding: 12px; border-bottom: 1px solid #ccc;">{event['confidence']:.2f}</td>
            <td style="padding: 12px; border-bottom: 1px solid #ccc;">{event['status']}</td>
        </tr>
        """
    return f"""
    <div style="max-height: 250px; overflow-y: auto; border: 1px solid #eee; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="background-color: #f8f9fa;">
                    <th style="padding: 12px; text-align: left; color: #666;">时间</th>
                    <th style="padding: 12px; text-align: left; color: #666;">置信度</th>
                    <th style="padding: 12px; text-align: left; color: #666;">状态</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """


def image_inference(image: np.ndarray) -> tuple:
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    alarm_status, annotated_frame = detector.process_frame(frame)
    return (
        cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
        generate_alert_html(alarm_status),
        update_history_html(detector.history)
    )

def video_inference(video_path: str) -> Iterator[tuple]:
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        alarm_status, annotated_frame = detector.process_frame(frame)
        yield (
            cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
            generate_alert_html(alarm_status),
            update_history_html(detector.history)
        )
    cap.release()

def webcam_inference(camera_device: int) -> Iterator[tuple]:
    cap = cv2.VideoCapture(camera_device)
    if not cap.isOpened():
        yield None, "<div style='color: #eb5757; background: #ffebe9; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(255,0,0,0.1);'>错误：无法打开摄像头设备，请检查连接！</div>", []
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        alarm_status, annotated_frame = detector.process_frame(frame)
        yield (
            cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
            generate_alert_html(alarm_status),
            update_history_html(detector.history)
        )
    cap.release()

def reset_system():
    detector.alarm_status = False
    pygame.mixer.music.stop()
    return "", "", "", update_history_html(detector.history)

# 自定义主题样式
with gr.Blocks(
    theme=gr.themes.Default(
        primary_hue="blue",
        secondary_hue="orange",
        neutral_hue="gray",
        font=[gr.themes.GoogleFont("Roboto"), "sans-serif"],
    )
) as app:
    app.css = """
    .gradio-container {
        background-color: #fafafa;
        padding: 20px;
        border-radius: 15px;
    }
    .gr-button {
        border-color: #4CAF50;
        background: linear-gradient(45deg, #4CAF50, #28A745);
        color: white;
        border-radius: 15px;
        width: 100%;
        height: 40px;
        font-size: 16px;
        margin-bottom: 10px;
    }
    .gr-button:hover {
        border-color: #2C7CE6;
        background: linear-gradient(45deg, #4CAF50, #28A745);
        transform: scale(1.05);
        transition: transform 0.2s;
    }
    .gr-form-label.subheading {
        font-size: 18px;
        color: #333;
        font-weight: bold;
    }
    .gr-slider-handle {
        background-color: #4CAF50;
        border-color: #4CAF50;
    }
    .gr-tooltip {
        background-color: #333;
        color: white;
        border-radius: 8px;
        padding: 8px;
    }
    """

    # 自定义网页标题
    gr.Markdown("""
    <div style="text-align: center; padding-bottom: 20px;">
        <h1 style="color: #2C7CE6; margin-bottom: 10px;">🚨 智能摔倒检测系统</h1>
        <p style="color: #666; font-size: 16px;">
            西南石油大学 冯诗楠 版权所有<br>
            📧 <a href="mailto:18398998558@163.com" style="color: #4CAF50;">18398998558@163.com</a>
        </p>
    </div>
    """)

    # 控制面板
    with gr.Row():
        with gr.Column(scale=3):
            camera_device = gr.Number(
                label="拍照设备号 📷",
                value=DEFAULT_CAMERA,
                precision=0,
                info="请输入摄像头设备的编号",
                interactive=True,
                elem_id="camera-device-input"
            )
            conf_threshold = gr.Slider(
                label="检测灵敏度 ⚖️",
                minimum=0.1,
                maximum=1.0,
                step=0.05,
                value=INIT_CONF_THRESHOLD,
                info="调整此值可以改变检测的灵敏度",
                interactive=True,
                elem_id="conf-slider"
            )
        with gr.Column(scale=2):
            with gr.Row():
                reset_btn = gr.Button("重置系统 🔄", variant="primary", size="lg")
                clear_history_btn = gr.Button("清除记录 ❌", variant="primary", size="lg")
            with gr.Row():
                save_history_btn = gr.Button("保存记录 💾", variant="primary", size="lg")
                auto_save_toggle = gr.Checkbox(
                    label="自动保存记录 🔄",
                    value=True,
                    info="勾选后系统会自动保存检测到的摔倒事件",
                    interactive=True,
                    elem_id="auto-save-toggle"
                )

    # 主显示区
    with gr.Tabs():
        with gr.Tab("图像检测 🖼️"):
            with gr.Row():
                img_input = gr.Image(label="输入图像", type="numpy", height=450)
                img_output = gr.Image(label="检测结果", height=450)
            alert_output_img = gr.HTML()
            with gr.Row():
                img_btn = gr.Button("开始检测 🚀", variant="primary", size="lg")
                img_clear_btn = gr.Button("清空 🧹", variant="stop", size="lg")

        with gr.Tab("视频检测 🎥"):
            with gr.Row():
                vid_input = gr.Video(label="输入视频", height=450)
                vid_output = gr.Image(label="检测画面", height=450)
            alert_output_vid = gr.HTML()
            with gr.Row():
                vid_btn = gr.Button("开始检测 🚀", variant="primary", size="lg")

        with gr.Tab("实时检测 🌐"):
            with gr.Row():
                webcam_output = gr.Image(
                    label="实时画面",
                    streaming=True,
                    height=450,
                    show_label=False,
                    interactive=False
                )
                alert_output_webcam = gr.HTML()
            with gr.Row():
                webcam_btn = gr.Button("启动摄像头 🔎", variant="primary", size="lg")

    # 历史记录
    history_html = gr.HTML(update_history_html(detector.history), label="事件历史 📂")

    # 事件绑定
    conf_threshold.change(
        fn=detector.set_conf_threshold,
        inputs=conf_threshold
    )
    reset_btn.click(
        fn=reset_system,
        outputs=[alert_output_img, alert_output_vid, alert_output_webcam, history_html]
    )
    clear_history_btn.click(
        fn=lambda: (detector.clear_history(), update_history_html([])),
        outputs=[history_html]
    )
    save_history_btn.click(
        fn=lambda: gr.Info("保存成功！" if detector.save_to_file() else "保存失败！"),
    )
    auto_save_toggle.change(
        fn=detector.toggle_auto_save,
        inputs=auto_save_toggle
    )
    img_btn.click(
        fn=image_inference,
        inputs=[img_input],
        outputs=[img_output, alert_output_img, history_html]
    )
    img_clear_btn.click(
        fn=lambda: (None, None, ""),
        outputs=[img_input, img_output, alert_output_img]
    )
    vid_btn.click(
        fn=video_inference,
        inputs=[vid_input],
        outputs=[vid_output, alert_output_vid, history_html]
    )
    webcam_btn.click(
        fn=webcam_inference,
        inputs=[camera_device],
        outputs=[webcam_output, alert_output_webcam, history_html]
    )

if __name__ == "__main__":
    app.launch(
        server_port=7860,
        show_error=True,
        favicon_path=None
    )
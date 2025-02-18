import cv2
import time
import numpy as np
import gradio as gr
from typing import Iterator
from ultralytics import YOLO
import pygame

import requests

# 初始化 pygame.mixer
pygame.mixer.init()

# ==== 系统配置 =========================
MODEL_PATH = "runs/detect/fall_detection_optimized/weights/best.pt"
FALL_CLASS_ID = 0
INIT_CONF_THRESHOLD = 0.5
TRIGGER_FRAMES = 15
DEFAULT_CAMERA = 0
ALARM_SOUND_PATH = r"C:\Users\AC_Chris\Desktop\alarm.wav"
HISTORY_FILE_PATH = r"C:\Users\AC_Chris\Desktop\history.txt"


# ======================================
class FallDetectionSystem:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.fall_counter = 0
        self.alarm_status = False
        self.alarm_start_time = 0
        self.alarm_duration = 10
        self.conf_threshold = INIT_CONF_THRESHOLD
        self.history = []
        self.last_message = ""

    def set_conf_threshold(self, threshold: float):
        self.conf_threshold = threshold

    def clear_history(self):
        self.history = []
        self.last_message = "历史记录已清空"
        return self.history, self.last_message

    def save_to_file(self) -> tuple[bool, str]:
        """手动保存历史记录到文件"""
        try:
            with open(HISTORY_FILE_PATH, "a", encoding="utf-8") as f:  # Append mode
                for event in self.history:
                    f.write(f"{event['time']} | 置信度: {event['confidence']:.2f} | 状态: {event['status']}\n")
            self.last_message = f"✅ 所有记录已成功保存至 {HISTORY_FILE_PATH}"
            return True, self.last_message
        except Exception as e:
            self.last_message = f"⚠️ 保存失败: {str(e)}"
            return False, self.last_message

    def process_frame(self, frame: np.ndarray) -> tuple:
        results = self.model(frame, verbose=False)[0]
        fall_detected = False
        annotated_frame = results.plot()

        # 检测循环
        for box in results.boxes:
            if int(box.cls) == FALL_CLASS_ID and float(box.conf) > self.conf_threshold:
                self.fall_counter += 1
                if self.fall_counter >= TRIGGER_FRAMES:
                    event_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    new_event = {
                        "time": event_time,
                        "confidence": float(box.conf),
                        "status": "fallen"
                    }
                    self.history.append(new_event)

                    if not self.alarm_status:
                        pygame.mixer.music.load(ALARM_SOUND_PATH)
                        pygame.mixer.music.play()
                        self.alarm_status = True
                        self.alarm_start_time = time.time()

                    fall_detected = True
                    self.fall_counter = 0
                break
        else:
            self.fall_counter = max(0, self.fall_counter - 1)

        # 报警超时处理
        if self.alarm_status and (time.time() - self.alarm_start_time) > self.alarm_duration:
            self.alarm_status = False
            pygame.mixer.music.stop()
            self.last_message = "⏰ 警报已自动解除"

        return self.alarm_status, annotated_frame, self.last_message

def send_sms(phone_number: str, message: str) -> str:
    """
    模拟短信发送（伪逻辑）
    """
    print("发送短信:")
    print(f"手机号码: {phone_number}")
    print(f"短信内容: {message}")
    detector.last_message = "短信已发送 (模拟)"
    return detector.last_message

def format_message(message: str) -> str:
    """增强的消息格式化函数"""
    if message:
        messages = message.split("|")
        items = "".join([f"<li>{msg.strip()}</li>" for msg in messages])
        return f"""
        <div style="
            background: #fff3cd;
            color: #856404;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #ffeeba;
            margin: 10px 0;
            font-size: 16px;
        ">
            <ul style="margin:0;padding-left:20px;">{items}</ul>
        </div>
        """
    return ""

def check_camera_availability(camera_device: int) -> str:
    if camera_device not in [0, 1]:
      return "🔴 无效的摄像头设备号，请选择 0 或 1"
    cap = cv2.VideoCapture(camera_device)
    is_available = cap.isOpened()
    cap.release()
    return "🟢 摄像头可用" if is_available else "🔴 摄像头未接入或未打开"

def update_history_html(history: list[dict]) -> str:
    rows = ""
    for event in history[-10:]:  # 只显示最后10条
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
                    <th style="padding: 12px; text-align: left;">时间</th>
                    <th style="padding: 12px; text-align: left;">置信度</th>
                    <th style="padding: 12px; text-align: left;">状态</th>
                </tr>
            </thead>
            <tbody>
                {rows if rows else '<tr><td colspan="3" style="text-align:center;">暂无记录</td></tr>'}
            </tbody>
        </table>
    </div>
    """

def image_inference(image: np.ndarray) -> tuple:
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    alarm_status, annotated_frame, message = detector.process_frame(frame)
    return (
        cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
        generate_alert_html(alarm_status),
        update_history_html(detector.history),
        format_message(message)
    )


def video_inference(video_path: str) -> Iterator[tuple]:
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        alarm_status, annotated_frame, message = detector.process_frame(frame)
        frame_count += 1
        # Increase frame rate by yielding more often
        if frame_count % 2 == 0:  # Yield every other frame for higher perceived frame rate
            yield (
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                generate_alert_html(alarm_status),
                update_history_html(detector.history),
                format_message(message)
            )
        if frame_count == total_frames:  # All frames processed
            yield (
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                generate_alert_html(alarm_status),
                update_history_html(detector.history),
                format_message("视频检测完成")
            )
    cap.release()


def webcam_inference(camera_device: int) -> Iterator[tuple]:
    if camera_device not in [0, 1]:
        yield None, "错误：摄像头设备号无效，请选择 0 (自带摄像头) 或 1 (外接摄像头)！", [], ""
        return

    cap = None  # 初始化 cap 为 None
    try:
        # 首先尝试 DirectShow (Windows)
        cap = cv2.VideoCapture(camera_device, cv2.CAP_DSHOW)
        if cap is None or not cap.isOpened():
            # 如果 DirectShow 失败，尝试 Video4Linux2 (Linux/macOS，也可能在Windows上工作)
            cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)

    except Exception as e:
        yield None, f"错误：打开摄像头设备 {camera_device} 失败：{e}", [], ""
        return

    if cap is None or not cap.isOpened():
        yield None, f"错误：无法打开摄像头设备 {camera_device}，请检查连接或设备号是否正确！", [], ""
        return

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                yield None, "错误：摄像头读取失败，请检查摄像头是否被占用或已断开连接！", [], ""
                break

            alarm_status, annotated_frame, message = detector.process_frame(frame)
            yield (
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                generate_alert_html(alarm_status),
                update_history_html(detector.history),
                format_message(message)
            )
    finally:
        if cap is not None:
            cap.release()

def generate_alert_html(alarm_status: bool) -> str:
    if alarm_status:
        return """
        <div style="
            background: #ff4444;
            color: white;
            padding: 20px;
            border-radius: 10px;
            font-size: 24px;
            text-align: center;
            animation: alarm 1s infinite;
            border: 2px solid #cc0000;
            margin: 10px 0;
        ">
            🚨 检测到摔倒！立即处理！
        </div>
        <style>
            @keyframes alarm {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
        </style>
        """
    return """
    <div style="
        background: #4CAF50;
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-size: 18px;
        text-align: center;
        margin: 10px 0;
    ">
        ✅ 系统运行正常
    </div>
    """
def mute_system():
    detector.alarm_status = False
    pygame.mixer.music.stop()
    detector.last_message = "已解除警报 🔇"
    gr.Info(detector.last_message)
    return "", "", "", update_history_html(detector.history)


def clear_history_and_show_message():
    history, message = detector.clear_history()
    gr.Info(message)  # 使用 gr.Info 显示消息
    return update_history_html([]), ""  # 返回 history_html 和 message_output 的值

def save_and_show_message():
    success, message = detector.save_to_file()
    gr.Info(message)
    return ""
def send_sms_and_show_message(phone_number):
    message = send_sms(phone_number=phone_number, message="检测到摔倒事件！")
    gr.Info(message)
    return ""

detector = FallDetectionSystem()

with gr.Blocks(
        theme=gr.themes.Default(
            primary_hue="blue",
            secondary_hue="orange",
            neutral_hue="gray",
            font=[gr.themes.GoogleFont("Roboto"), "sans-serif"],
        )
)as app:
    # 使用 gr.Markdown 和 HTML 标签来居中内容
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h1 style="margin: 20px 0;">🚨 智能摔倒检测系统</h1>
            <p style="margin-bottom: 20px;">
                西南石油大学 冯诗楠 版权所有 <br>
                📧 18398998558@163.com
            </p>
        </div>
        """
    )

    # 操作指南部分
    with gr.Accordion("📚 操作指南", open=False):
        gr.Markdown(
            """
            **系统使用说明：**
            1. 📷 摄像头设置：连接设备后输入正确编号，自带摄像头编号为0，外接摄像头编号为1
            2. ⚖️ 灵敏度调节：根据需要调整检测阈值，检测阈值越低，越容易检测出人员摔倒
            3. 🚀 启动检测：选择对应模式后点击检测按钮
            4. 📊 历史记录：自动保存最新10条记录
            5. 🔕 解除警报，可关闭警报声和页面的警报通知
            6. 📱 发送短信警报，连接阿里云短信服务，点击后可发送短信通知相关人员
            """
        )

    with gr.Row():
        with gr.Column(scale=3):
            camera_device = gr.Number(
                label="拍照设备号 📷",
                value=0,  # 默认值为 0
                precision=0,
                info="请选择 0 (自带摄像头) 或 1 (外接摄像头)",
                interactive=True,
                minimum=0,  # 限制最小值
                maximum=1  # 限制最大值
            )
            camera_status = gr.HTML(check_camera_availability(DEFAULT_CAMERA))
            conf_threshold = gr.Slider(
                label="检测灵敏度 ⚖️",
                minimum=0.1,
                maximum=1.0,
                step=0.05,
                value=INIT_CONF_THRESHOLD,
                info="调整此值可以改变检测的灵敏度",
                interactive=True
            )
        with gr.Column(scale=2):
            mute_btn = gr.Button("解除警报 🔕", variant="primary", size="lg")
            clear_history_btn = gr.Button("清除记录 ❌", variant="primary", size="lg")
            manual_save_btn = gr.Button("手动保存记录 💾", variant="primary", size="lg")
            send_sms_btn = gr.Button("发送短信警报 📱", variant="primary", size="lg")
            phone_number_input = gr.Textbox(
                label="接收短信的手机号码",
                placeholder="请输入手机号码",
                interactive=True
            )

    with gr.Tabs():
        with gr.Tab("图像检测 🖼️"):
            with gr.Row():
                img_input = gr.Image(label="输入图像", type="numpy", height=450)
                img_output = gr.Image(label="检测结果", height=450)
            alert_output_img = gr.HTML()
            message_output_img = gr.HTML()
            with gr.Row():
                img_btn = gr.Button("开始检测 🚀", variant="primary", size="lg")
                img_clear_btn = gr.Button("清空 🧹", variant="stop", size="lg")

        with gr.Tab("视频检测 🎥"):
            with gr.Row():
                vid_input = gr.Video(label="输入视频", height=450)
                vid_output = gr.Image(label="检测画面", height=450)
            alert_output_vid = gr.HTML()
            message_output_vid = gr.HTML()
            with gr.Row():
                vid_btn = gr.Button("开始检测 🚀", variant="primary", size="lg")
                vid_clear_btn = gr.Button("清空 🧹", variant="stop", size="lg")

        with gr.Tab("实时检测 🌐"):
            with gr.Column():
                webcam_output = gr.Image(
                    label="实时画面",
                    streaming=True,
                    height=450,  # 设置固定高度
                    width=640  # 设置固定宽度
                )
                alert_output_webcam = gr.HTML()
                message_output_webcam = gr.HTML()
            with gr.Row():
                webcam_btn = gr.Button("启动摄像头 🔎", variant="primary", size="lg")

    history_html = gr.HTML(update_history_html(detector.history), label="最近事件记录")
    message_output = gr.HTML()

    # 事件绑定
    conf_threshold.change(fn=detector.set_conf_threshold, inputs=conf_threshold)
    mute_btn.click(
        fn=mute_system,
        outputs=[alert_output_img, alert_output_vid, alert_output_webcam, history_html]
    )
    clear_history_btn.click(fn=clear_history_and_show_message, outputs=[history_html, message_output])
    manual_save_btn.click(fn=save_and_show_message, outputs=[message_output])
    send_sms_btn.click(fn=send_sms_and_show_message, inputs=[phone_number_input], outputs=[message_output])
    img_btn.click(
        fn=image_inference,
        inputs=[img_input],
        outputs=[img_output, alert_output_img, history_html, message_output_img]
    )
    img_clear_btn.click(
        fn=lambda: (None, None, "", ""),
        outputs=[img_input, img_output, alert_output_img, message_output_img]
    )
    vid_btn.click(
        fn=video_inference,
        inputs=[vid_input],
        outputs=[vid_output, alert_output_vid, history_html, message_output_vid]
    )
    vid_clear_btn.click(
        fn=lambda: (None, None, "", ""),
        outputs=[vid_input, vid_output, alert_output_vid, message_output_vid]
    )
    webcam_btn.click(
        fn=webcam_inference,
        inputs=[camera_device],
        outputs=[webcam_output, alert_output_webcam, history_html, message_output_webcam]
    )
    camera_device.change(fn=check_camera_availability, inputs=camera_device, outputs=camera_status)

if __name__ == "__main__":
    app.launch(
        server_port=7860,
        show_error=True,
        favicon_path=None
    )
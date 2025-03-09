import cv2
from dotenv import load_dotenv
import os
import time
import numpy as np
import gradio as gr
from typing import Iterator
from ultralytics import YOLO
import pygame
import json
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest

# Initialize pygame.mixer for alarm sound
pygame.mixer.init()
load_dotenv()

# ==== System Configuration =========================
MODEL_PATH = "runs/detect/fall_detection_optimized/weights/best.pt"
FALL_CLASS_ID = 0
INIT_CONF_THRESHOLD = 0.5
TRIGGER_FRAMES = 15
DEFAULT_CAMERA = 0
ALARM_SOUND_PATH = r"C:\Users\AC_Chris\Desktop\alarm.wav"
HISTORY_FILE_PATH = r"C:\Users\AC_Chris\Desktop\history.txt"
TARGET_WIDTH = 1280
TARGET_HEIGHT = 480
SKIP_FRAMES = 2
webcam_running = False
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

    def get_last_fall_time(self):
        return self.history[-1]["time"] if self.history else None

    def clear_history(self):
        self.history = []
        self.last_message = ""
        return self.history

    def save_to_file(self) -> tuple[bool, str]:
        try:
            with open(HISTORY_FILE_PATH, "a", encoding="utf-8") as f:
                for event in self.history:
                    f.write(f"{event['time']} | 置信度: {event['confidence']:.2f} | 状态: {event['status']}\n")
            self.last_message = ""
            return True, ""
        except Exception as e:
            self.last_message = ""
            return False, ""

    def process_frame(self, frame: np.ndarray) -> tuple:
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        annotated_frame = results[0].plot()
        for box in results[0].boxes:
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
                    self.fall_counter = 0
                break
        else:
            self.fall_counter = max(0, self.fall_counter - 1)
        if self.alarm_status and (time.time() - self.alarm_start_time) > self.alarm_duration:
            self.alarm_status = False
            pygame.mixer.music.stop()
            self.last_message = ""
        return self.alarm_status, annotated_frame, ""

def send_sms(phone_number: str, event_time: str) -> str:
    try:
        client = AcsClient(
            os.getenv("ALIYUN_KEY_ID"),
            os.getenv("ALIYUN_KEY_SECRET"),
            "cn-chengdu"
        )
        request = CommonRequest()
        request.set_accept_format('json')
        request.set_domain('dysmsapi.aliyuncs.com')
        request.set_method('POST')
        request.set_protocol_type('https')
        request.set_version('2017-05-25')
        request.set_action_name('SendSms')
        request.add_query_param('PhoneNumbers', phone_number)
        request.add_query_param('SignName', "智能跌倒检测系统")
        request.add_query_param('TemplateCode', "SMS_xxxxxxxxx")
        request.add_query_param('TemplateParam', json.dumps({"time": event_time}))
        response = client.do_action_with_exception(request)
        response_data = json.loads(response.decode('utf-8'))
        if response_data.get('Code') == 'OK':
            return "✅ 短信已成功发送"
        return f"⚠️ 短信发送失败：{response_data.get('Message', '未知错误')}"
    except Exception as e:
        return f"⚠️ 短信发送异常：{str(e)}"

def format_message(message: str) -> str:
    if message:
        messages = message.split(" |")
        items = "".join([f" {msg.strip()}<br>" for msg in messages])
        return f'<div style="padding: 10px; background: #f0f0f0; border-radius: 5px;">{items}</div>'
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
    for event in history[-10:]:
        rows += f"""
            <tr>
                <td>{event['time']}</td>
                <td>{event['confidence']:.2f}</td>
                <td>{event['status']}</td>
            </tr>
        """
    return f"""
        <table style="width:100%; border-collapse: collapse;">
            <thead>
                <tr>
                    <th>时间</th>
                    <th>置信度</th>
                    <th>状态</th>
                </tr>
            </thead>
            <tbody>
                {rows if rows else '<tr><td colspan="3">暂无记录</td></tr>'}
            </tbody>
        </table>
    """

def image_inference(image: np.ndarray) -> tuple:
    detector.last_message = ""  # 重置消息缓存
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    alarm_status, annotated_frame, _ = detector.process_frame(frame)
    return (
        cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
        generate_alert_html(alarm_status),
        update_history_html(detector.history),
        ""  # 强制返回空消息
    )

def video_inference(video_path: str) -> Iterator[tuple]:
    detector.last_message = ""  # 重置消息缓存
    cap = cv2.VideoCapture(video_path)
    frame_count = 0  # 新增帧计数器
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % (SKIP_FRAMES + 1) != 0:  # 跳帧逻辑
            continue
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        alarm_status, annotated_frame, _ = detector.process_frame(frame)
        yield (
            cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
            generate_alert_html(alarm_status),
            update_history_html(detector.history),
            ""
        )
    cap.release()

def webcam_inference(camera_device: int) -> Iterator[tuple]:
    global webcam_running
    detector.last_message = ""  # 重置消息缓存
    cap = cv2.VideoCapture(camera_device)
    webcam_running = True
    while webcam_running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        alarm_status, annotated_frame, _ = detector.process_frame(frame)
        yield (
            cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
            generate_alert_html(alarm_status),
            update_history_html(detector.history),
            ""  # 强制返回空消息
        )
    cap.release()

def generate_alert_html(alarm_status: bool) -> str:
    if not alarm_status:
        return f'''
        <div style="
            padding: 20px 40px;  /* 增加内边距 */
            background: #e8f5e9;
            border: 3px solid #43a047;  /* 加粗边框 */
            border-radius: 10px;
            color: #2e7d32;
            font-size: 24px;  /* 加大文字 */
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
        ">
            <span style="font-size: 36px;">✅</span>  <!-- 加大图标 -->
            <span>系统运行正常</span>
        </div>
        '''
    return f'''
    <div style="
        padding: 20px 40px;
        background: #ffebee;
        border: 3px solid #c62828;
        border-radius: 10px;
        color: #b71c1c;
        font-size: 24px;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 15px;
        animation: pulse 1s infinite;
    ">
        <span style="font-size: 36px;">🚨</span>
        <span>检测到摔倒！立即处理！</span>
    </div>
    <style>
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}  /* 加大脉冲效果 */
            100% {{ transform: scale(1); }}
        }}
    </style>
    '''

def mute_system():
    detector.last_message = ""  # 重置消息缓存
    if not detector.history:
        gr.Info("⚠️ 无近期摔倒记录")
    else:
        detector.alarm_status = False
        pygame.mixer.music.stop()
        gr.Info("已解除警报 🔇")
    return (
        generate_alert_html(detector.alarm_status),
        generate_alert_html(detector.alarm_status),
        generate_alert_html(detector.alarm_status),
        update_history_html(detector.history)
    )

def clear_history_and_show_message():
    detector.clear_history()
    gr.Info("历史记录已清空")
    return update_history_html(detector.history), "", "", ""

def save_and_show_message():
    success, _ = detector.save_to_file()
    if not detector.history:
        gr.Info("⚠️ 无近期摔倒记录")
    else:
        gr.Info("✅ 记录保存成功" if success else "⚠️ 保存失败")
    return "", "", ""

def send_sms_and_show_message(phone_number):
    detector.last_message = ""  # 重置消息缓存
    if not detector.history:
        gr.Info("⚠️ 无近期摔倒记录")
        return "", "", ""
    last_time = detector.get_last_fall_time()
    if not last_time:
        gr.Info("⚠️ 无近期摔倒记录")
        return "", "", ""
    if not phone_number.isdigit() or len(phone_number) != 11:
        gr.Info("⚠️ 手机号码格式错误")
        return "", "", ""
    message = send_sms(phone_number=phone_number, event_time=last_time)
    gr.Info(message)
    return "", "", ""

def stop_webcam():
    global webcam_running
    webcam_running = False
    return (
        None,
        generate_alert_html(False),
        update_history_html(detector.history),
        ""
    )

detector = FallDetectionSystem()

with gr.Blocks(
    theme=gr.themes.Default(
        primary_hue="blue",
        secondary_hue="orange",
        neutral_hue="gray",
        font=[gr.themes.GoogleFont("Roboto"), "sans-serif"],
    ),
    css='''
    .centered-image {
        display: block;
        margin: 0 auto;
        width: 100%;
    }
    '''
) as app:
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
    with gr.Accordion("📚 操作指南", open=False):
        gr.Markdown(
            """
            **系统使用说明：**
            1. 📷 摄像头设置：连接设备后输入正确编号，自带摄像头编号为0，外接摄像头编号为1
            2. ⚖️ 灵敏度调节：根据需要调整检测阈值，检测阈值越低，越容易检测出人员摔倒
            3. 🚀 启动检测：选择对应模式后点击检测按钮
            4. 📊 历史记录：自动保存最新10条记录
            5. 🔕 解除警报，可关闭警报声和页面的警报通知
            6. 📱 发送短信警报，连接阿里云短信服务 ，点击后可发送短信通知相关人员
            """
        )
    with gr.Row():
        with gr.Column(scale=3):
            camera_device = gr.Number(
                label="拍照设备号 📷",
                value=DEFAULT_CAMERA,
                precision=0,
                info="请选择 0 (自带摄像头) 或 1 (外接摄像头)",
                interactive=True,
                minimum=0,
                maximum=1
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
                    height=450,
                    width=640,
                    elem_classes="centered-image"
                )
                alert_output_webcam = gr.HTML()
                message_output_webcam = gr.HTML()
            with gr.Row():
                webcam_btn = gr.Button("启动摄像头 🔎", variant="primary", size="lg")
                close_webcam_btn = gr.Button("关闭摄像头 🛑", variant="stop", size="lg")
    history_html = gr.HTML(update_history_html(detector.history), label="最近事件记录")
    message_output = gr.HTML()
    conf_threshold.change(fn=detector.set_conf_threshold, inputs=conf_threshold)
    mute_btn.click(
        fn=mute_system,
        outputs=[alert_output_img, alert_output_vid, alert_output_webcam, history_html]
    )
    clear_history_btn.click(
        fn=clear_history_and_show_message,
        outputs=[history_html, message_output_img, message_output_vid, message_output_webcam]
    )
    manual_save_btn.click(
        fn=save_and_show_message,
        outputs=[message_output_img, message_output_vid, message_output_webcam]
    )
    send_sms_btn.click(
        fn=send_sms_and_show_message,
        inputs=[phone_number_input],
        outputs=[message_output_img, message_output_vid, message_output_webcam]
    )
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
    close_webcam_btn.click(
        fn=stop_webcam,
        outputs=[webcam_output, alert_output_webcam, history_html, message_output_webcam]
    )
    camera_device.change(fn=check_camera_availability, inputs=camera_device, outputs=camera_status)

if __name__ == "__main__":
    app.launch(
        server_port=7860,
        show_error=True,
        favicon_path=None
    )
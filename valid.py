from ultralytics import YOLO
import argparse
import matplotlib.pyplot as plt

# 设置 Matplotlib 字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def validate_model(
    model_weights,  # 训练好的模型权重路径
    config_path,    # data.yaml 路径
    batch=8,        # 批量大小
    device="0",     # 使用的设备
    imgsz=640,      # 输入图像尺寸
    name="val_results"  # 验证结果保存的目录名称
):
    """
    使用验证集验证训练好的 YOLOv8 模型
    Args:
        model_weights (str): 训练好的模型权重路径（如 best.pt）
        config_path (str): data.yaml 路径
        batch (int): 批量大小
        device (str): 使用的设备
        imgsz (int): 输入图像尺寸
        name (str): 验证结果保存的目录名称
    """
    # 加载训练好的模型
    model = YOLO(model_weights)

    # 开始验证
    results = model.val(
        data=config_path,  # 数据集配置文件路径
        batch=batch,       # 批量大小
        device=device,     # 使用的设备
        imgsz=imgsz,       # 输入图像尺寸
        name=name,         # 验证结果保存的目录名称
        split="val",       # 使用验证集（默认是 val）
        save_json=True,    # 保存 JSON 格式的验证结果
        save_hybrid=True,  # 保存混合格式的验证结果
        conf=0.001,        # 置信度阈值
        iou=0.6,           # IoU 阈值
    )

    # 打印验证结果
    print("验证结果：")
    print(f"mAP50: {results.box.map50}")
    print(f"mAP50-95: {results.box.map}")
    print(f"Precision: {results.box.p[0]}")  # 使用 results.box.p 访问 Precision
    print(f"Recall: {results.box.r[0]}")     # 使用 results.box.r 访问 Recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate YOLOv8 Model")
    parser.add_argument("--model-weights", type=str, default="runs/detect/fall_detection_optimized/weights/best.pt", help="训练好的模型权重路径")
    parser.add_argument("--config", type=str, default="D:/ultralytics-main/ultralytics/dataset/data.yaml", help="data.yaml 路径")
    parser.add_argument("--batch", type=int, default=8, help="批量大小")
    parser.add_argument("--device", type=str, default="0", help="GPU 设备")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--name", type=str, default="val_results", help="验证结果保存的目录名称")
    args = parser.parse_args()

    # 调用验证函数
    validate_model(
        model_weights=args.model_weights,
        config_path=args.config,
        batch=args.batch,
        device=args.device,
        imgsz=args.imgsz,
        name=args.name
    )
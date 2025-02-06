from ultralytics import YOLO
import argparse

def train_model(
    config_path,
    model_config="yolov8s.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device="0",
    augment=True,
    lr0=0.01,
    patience=15,
    save_period=10
):
    """
    优化后的摔倒检测训练函数（适配 RTX 2060 6GB）
    Args:
        config_path (str): data.yaml 路径
        model_config (str): 模型配置文件路径（默认 yolov8s.yaml）
        epochs (int): 训练轮次
        imgsz (int): 输入图像尺寸
        batch (int): 批量大小
        device (str): 使用的设备
        augment (bool): 是否启用数据增强
        lr0 (float): 初始学习率
        patience (int): 早停耐心值
        save_period (int): 模型保存周期
    """
    # 加载模型结构（从头训练）
    model = YOLO(model_config)

    # 开始训练
    model.train(
        data=config_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        augment=augment,
        lr0=lr0,
        patience=patience,
        save_period=save_period,
        name="fall_detection_optimized",
        exist_ok=True,
        # 显存优化选项（YOLOv8 默认启用混合精度 FP16）
        cos_lr=True,        # 启用余弦学习率调度
        fliplr=0.2,         # 水平翻转概率（降低增强开销）
        flipud=0.1,         # 垂直翻转概率
        degrees=10,         # 旋转角度范围（降低计算量）
        mosaic=0.0,         # 禁用 Mosaic 增强（显存杀手）
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized Training for Fall Detection")
    parser.add_argument("--config", type=str, default="D:/ultralytics-main/ultralytics/dataset/data.yaml", help="data.yaml 路径")
    parser.add_argument("--model-config", type=str, default="yolov8s.yaml", help="模型配置文件路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮次")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--batch", type=int, default=8, help="批量大小")
    parser.add_argument("--device", type=str, default="0", help="GPU 设备")
    parser.add_argument("--augment", action="store_true", help="启用数据增强")
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    parser.add_argument("--patience", type=int, default=15, help="早停耐心值")
    parser.add_argument("--save-period", type=int, default=10, help="模型保存周期")
    args = parser.parse_args()

    # 调用训练函数
    train_model(
        config_path=args.config,
        model_config=args.model_config,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        augment=args.augment,
        lr0=args.lr0,
        patience=args.patience,
        save_period=args.save_period
    )
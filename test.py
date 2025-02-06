from ultralytics import YOLO

def main():
    # 加载训练好的模型
    model = YOLO("runs/detect/fall_detection_optimized/weights/best.pt")

    # 在测试集上评估
    results = model.val(
        data="D:/ultralytics-main/ultralytics/dataset/data.yaml",
        split="test",  # 使用测试集
        batch=8,
        device="0",
        imgsz=640,
        name="test_results"
    )

    # 输出关键指标
    print(f"测试集 mAP50: {results.box.map50:.4f}")
    print(f"测试集 mAP50-95: {results.box.map:.4f}")
    print(f"测试集 Precision: {results.box.p[0]:.4f}")
    print(f"测试集 Recall: {results.box.r[0]:.4f}")

if __name__ == "__main__":
    # 确保主模块代码被保护
    main()
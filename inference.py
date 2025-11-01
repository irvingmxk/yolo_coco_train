#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Inference Script
用于使用训练好的模型进行推理
"""

from pathlib import Path
from ultralytics import YOLO
import cv2

def predict_image(model_path, image_path, save_dir='runs/predict'):
    """
    对单张图片进行预测
    """
    model = YOLO(model_path)
    
    results = model.predict(
        source=image_path,
        save=True,
        save_txt=True,
        save_conf=True,
        project=save_dir,
        name='results',
        exist_ok=True,
        conf=0.25,
        iou=0.7,
        max_det=300,
        show_labels=True,
        show_conf=True,
        line_width=2,
    )
    
    return results

def predict_folder(model_path, folder_path, save_dir='runs/predict'):
    """
    对文件夹中的所有图片进行预测
    """
    model = YOLO(model_path)
    
    results = model.predict(
        source=folder_path,
        save=True,
        save_txt=True,
        save_conf=True,
        project=save_dir,
        name='results',
        exist_ok=True,
        conf=0.25,
        iou=0.7,
        max_det=300,
        show_labels=True,
        show_conf=True,
        line_width=2,
    )
    
    return results

def main():
    """
    主函数
    """
    # 模型路径（按优先级尝试）
    model_paths = [
        "/workspace/yolo/runs/train/bubble_optimized/weights/best.pt"
        # "runs/train/bubble_detection/weights/best.pt"
        # "runs/train/yolov8s_bubble/weights/best.pt",
        # 'runs/train/yolo11n_bubble/weights/best.pt',  # YOLO11n训练的模型
        # "runs/train/yolo11s_bubble/weights/best.pt",
        # "runs/train/yolo11m_bubble/weights/best.pt",
    ]
    
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            break
    
    if model_path is None:
        print("❌ 错误: 未找到训练好的模型")      
        print("\n请先训练模型:")
        print("  python train_yolo11m.py")
        print("\n或指定模型路径:")
        print("  例如: python inference.py --model runs/train/yolo11m_bubble/weights/best.pt")
        return
    
    # 测试图片路径（使用验证集的图片）
    test_image = '/workspace/yolo/image'
    # test_image = '/workspace/yolo/bumble.jpg'
    
    if not Path(test_image).exists():
        print(f"❌ 错误: 测试图片路径不存在: {test_image}")
        return
    
    print(f"使用模型: {model_path}\n")
    
    if Path(test_image).is_dir():
        print(f"对文件夹进行预测: {test_image}")
        results = predict_folder(model_path, test_image)
    else:
        print(f"对单张图片进行预测: {test_image}")
        results = predict_image(model_path, test_image)
    
    print(f"\n✅ 预测完成！")
    print(f"结果已保存至: runs/predict/results")

if __name__ == '__main__':
    main()


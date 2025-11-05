#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Inference Script
用于使用训练好的模型进行推理
"""

from pathlib import Path
from ultralytics import YOLO
import cv2
import sys
import os

# 导入配置参数（从yolo_cueq模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'yolo_cueq'))
try:
    from config import CONFIG
except ImportError:
    # 如果导入失败，使用默认值
    CONFIG = {
        'yolo_conf': 0.5,
        'yolo_iou': 0.45,
    }
    print("⚠️  无法导入config.py，使用默认配置参数")

def predict_image(model_path, image_path, save_dir='runs/predict', conf=None, iou=None):
    """
    对单张图片进行预测
    
    Args:
        model_path: 模型路径
        image_path: 图片路径
        save_dir: 保存目录
        conf: 置信度阈值（默认使用config中的值）
        iou: IOU阈值（默认使用config中的值）
    """
    model = YOLO(model_path)
    
    # 使用config中的默认值，或传入的参数
    conf_threshold = conf if conf is not None else CONFIG['yolo_conf']
    iou_threshold = iou if iou is not None else CONFIG['yolo_iou']
    
    results = model.predict(
        source=image_path,
        save=True,
        save_txt=True,
        save_conf=True,
        project=save_dir,
        name='results',
        exist_ok=True,
        conf=conf_threshold,      # 使用config中的值：0.5
        iou=iou_threshold,        # 使用config中的值：0.45
        max_det=300,
        show_labels=True,
        show_conf=True,
        line_width=2,
    )
    
    return results

def predict_folder(model_path, folder_path, save_dir='runs/predict', conf=None, iou=None):
    """
    对文件夹中的所有图片进行预测
    
    Args:
        model_path: 模型路径
        folder_path: 文件夹路径
        save_dir: 保存目录
        conf: 置信度阈值（默认使用config中的值）
        iou: IOU阈值（默认使用config中的值）
    """
    model = YOLO(model_path)
    
    # 使用config中的默认值，或传入的参数
    conf_threshold = conf if conf is not None else CONFIG['yolo_conf']
    iou_threshold = iou if iou is not None else CONFIG['yolo_iou']
    
    results = model.predict(
        source=folder_path,
        save=True,
        save_txt=True,
        save_conf=True,
        project=save_dir,
        name='results',
        exist_ok=True,
        conf=conf_threshold,      # 使用config中的值：0.5
        iou=iou_threshold,        # 使用config中的值：0.45
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
        # "runs/train/yolo11m_bubble_251105/weights/best.pt"
        "/workspace/yolo/runs/train/yolo11s_bubble/weights/best.pt"
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
    
    print(f"使用模型: {model_path}")
    print(f"检测参数 (来自config.py):")
    print(f"  置信度阈值 (conf): {CONFIG['yolo_conf']}")
    print(f"  IOU阈值 (iou): {CONFIG['yolo_iou']}")
    print()
    
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


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Training Script
用于训练气泡检测模型
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml

# 配置参数
CONFIG = {
    'model': 'yolo11n.pt',  # 使用YOLO11n模型（更快更轻量）
    'epochs': 100,
    'batch': 16,
    'imgsz': 640,
    'device': 0,  # GPU设备编号，使用 'cpu' 进行CPU训练
    'workers': 8,
    'patience': 50,  # 早停patience
    'save_period': 10,  # 每N个epoch保存一次
    'val_split': 0.2,  # 验证集比例
}

def split_dataset(data_dir, val_split=0.2):
    """
    将数据集分割为训练集和验证集
    """
    import random
    
    data_dir = Path(data_dir)
    images_dir = data_dir / 'images'
    labels_dir = data_dir / 'labels'
    
    # 获取所有图片文件
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    image_files = sorted(image_files)
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 随机打乱
    random.seed(42)
    random.shuffle(image_files)
    
    # 分割
    val_size = int(len(image_files) * val_split)
    train_files = image_files[val_size:]
    val_files = image_files[:val_size]
    
    print(f"训练集: {len(train_files)} 张")
    print(f"验证集: {len(val_files)} 张")
    
    # 创建目录结构
    for split in ['train', 'val']:
        (data_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (data_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 复制文件到对应目录
    def copy_files(file_list, split_name):
        for img_path in file_list:
            label_path = labels_dir / (img_path.stem + '.txt')
            
            # 复制图片
            shutil.copy(img_path, data_dir / split_name / 'images' / img_path.name)
            
            # 复制标签（如果存在）
            if label_path.exists():
                shutil.copy(label_path, data_dir / split_name / 'labels' / label_path.name)
    
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    
    print("数据集分割完成！")
    return len(train_files), len(val_files)

def create_data_yaml(data_dir, class_names):
    """
    创建YOLO数据配置文件
    """
    data_dir = Path(data_dir).absolute()
    
    data_config = {
        'path': str(data_dir),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = data_dir / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, allow_unicode=True, sort_keys=False)
    
    print(f"配置文件已创建: {yaml_path}")
    return yaml_path

def train_yolo(data_yaml, config):
    """
    训练YOLO模型
    """
    # 加载模型
    model = YOLO(config['model'])
    
    print(f"\n开始训练...")
    print(f"模型: {config['model']}")
    print(f"训练轮数: {config['epochs']}")
    print(f"批次大小: {config['batch']}")
    print(f"图片大小: {config['imgsz']}")
    
    # 训练模型
    results = model.train(
        data=str(data_yaml),
        epochs=config['epochs'],
        batch=config['batch'],
        imgsz=config['imgsz'],
        device=config['device'],
        workers=config['workers'],
        patience=config['patience'],
        save_period=config['save_period'],
        project='runs/train',
        name='bubble_detection',
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=True,
        fraction=1.0,
        profile=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True,
        save=True,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        show_labels=True,
        show_conf=True,
        max_det=300,
        vid_stride=1,
        line_width=None,
        visualize=False,
        augment=False,
        agnostic_nms=False,
        retina_masks=False,
    )
    
    print("\n训练完成！")
    return results

def validate_model(model_path, data_yaml):
    """
    验证模型性能
    """
    model = YOLO(model_path)
    
    print(f"\n验证模型...")
    results = model.val(data=str(data_yaml))
    
    print(f"\n验证结果:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    
    return results

def main():
    """
    主函数
    """
    # 数据目录
    data_dir = Path('/workspace/yolo/data')
    
    # 读取类别
    classes_file = data_dir / 'classes.txt'
    with open(classes_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f if line.strip()]
    
    print(f"检测类别: {class_names}")
    print(f"类别数量: {len(class_names)}")
    
    # 检查是否已经分割数据集
    if not (data_dir / 'train').exists():
        print("\n正在分割数据集...")
        split_dataset(data_dir, val_split=CONFIG['val_split'])
    else:
        print("\n数据集已分割，跳过分割步骤")
    
    # 创建配置文件
    data_yaml = create_data_yaml(data_dir, class_names)
    
    # 训练模型
    results = train_yolo(data_yaml, CONFIG)
    
    # 获取最佳模型路径
    best_model = 'runs/train/bubble_detection/weights/best.pt'
    
    if os.path.exists(best_model):
        print(f"\n最佳模型已保存至: {best_model}")
        
        # 验证最佳模型
        validate_model(best_model, data_yaml)
    else:
        print("\n未找到最佳模型文件")

if __name__ == '__main__':
    main()

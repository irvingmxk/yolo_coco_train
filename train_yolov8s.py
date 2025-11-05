#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11s 气泡检测微调脚本
使用 YOLO11s 模型进行气泡检测的迁移学习训练
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch

# ============= 训练配置参数 =============
CONFIG = {
    # 模型配置
    'model': 'yolov8s.pt',          # YOLO8s 预训练模型
    
    # 训练参数
    'epochs': 100,                   # 训练轮数
    'batch': 16,                     # 批次大小（提升到16以获得更稳定训练）
    'imgsz': 640,                    # 输入图像大小
    'device': 0,                     # GPU设备（0,1,2... 或 'cpu'）
    'workers': 8,                    # 数据加载线程数
    
    # 优化器配置（使用YOLO官方推荐配置）
    'optimizer': 'auto',             # 优化器：auto=SGD（官方推荐）
    'lr0': 0.01,                     # 初始学习率
    'lrf': 0.01,                     # 最终学习率 (lr0 * lrf)
    'momentum': 0.937,               # SGD动量
    'weight_decay': 0.0005,          # 权重衰减
    
    # 训练策略
    'patience': 20,                  # 早停等待轮数（减少等待时间）
    'save_period': 10,               # 每N轮保存一次模型
    'cos_lr': False,                 # 使用线性学习率调度（更稳定）
    'warmup_epochs': 3,              # 预热轮数
    'warmup_momentum': 0.8,          # 预热初始动量
    'warmup_bias_lr': 0.1,           # 预热偏置学习率
    
    # 数据增强
    'hsv_h': 0.015,                  # HSV-色调增强
    'hsv_s': 0.7,                    # HSV-饱和度增强
    'hsv_v': 0.4,                    # HSV-亮度增强
    'degrees': 0.0,                  # 旋转角度 (+/- deg)
    'translate': 0.1,                # 平移 (+/- fraction)
    'scale': 0.5,                    # 缩放增益
    'shear': 0.0,                    # 剪切角度 (+/- deg)
    'perspective': 0.0,              # 透视变换
    'flipud': 0.0,                   # 上下翻转概率
    'fliplr': 0.5,                   # 左右翻转概率
    'mosaic': 1.0,                   # Mosaic增强概率
    'mixup': 0.0,                    # Mixup增强概率
    'copy_paste': 0.0,               # Copy-paste增强概率
    
    # 验证和保存
    'val': True,                     # 训练过程中验证
    'save': True,                    # 保存训练检查点
    'plots': True,                   # 保存绘图
    'save_json': False,              # 保存COCO JSON格式结果
    
    # 输出配置
    'project': 'runs/train',         # 项目目录
    'name': 'yolov8s_bubble',        # 实验名称
    'exist_ok': True,                # 覆盖已存在的实验
}


def check_environment():
    """
    检查运行环境
    """
    print("=" * 60)
    print("环境检查")
    print("=" * 60)
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA可用")
        print(f"   GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        print(f"   当前GPU: {torch.cuda.current_device()}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print(f"⚠️  CUDA不可用，将使用CPU训练（速度较慢）")
        CONFIG['device'] = 'cpu'
    
    # 检查模型文件
    model_path = Path(CONFIG['model'])
    if model_path.exists():
        print(f"✅ 模型文件存在: {model_path}")
    else:
        print(f"⚠️  模型文件不存在: {model_path}")
        print(f"   首次运行将自动下载预训练模型")
    
    # 检查数据集
    data_dir = Path('/workspace/yolo/data_1105')
    train_dir = data_dir / 'train' / 'images'
    val_dir = data_dir / 'val' / 'images'
    data_yaml = data_dir / 'data.yaml'
    
    if not data_yaml.exists():
        print(f"❌ 数据配置文件不存在: {data_yaml}")
        return False
    
    print(f"✅ 数据配置: {data_yaml}")
    
    if train_dir.exists():
        train_images = list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.png'))
        print(f"✅ 训练集: {len(train_images)} 张图片")
    else:
        print(f"❌ 训练集目录不存在: {train_dir}")
        return False
    
    if val_dir.exists():
        val_images = list(val_dir.glob('*.jpg')) + list(val_dir.glob('*.png'))
        print(f"✅ 验证集: {len(val_images)} 张图片")
    else:
        print(f"❌ 验证集目录不存在: {val_dir}")
        return False
    
    print("=" * 60)
    return True


def train_model(config):
    """
    训练YOLO8s模型
    """
    print("\n" + "=" * 60)
    print("开始训练 YOLO8s")
    print("=" * 60)
    
    # 打印配置
    print("\n训练配置:")
    print(f"  模型: {config['model']}")
    print(f"  训练轮数: {config['epochs']}")
    print(f"  批次大小: {config['batch']}")
    print(f"  图像大小: {config['imgsz']}")
    print(f"  设备: {config['device']}")
    print(f"  优化器: {config['optimizer']}")
    print(f"  初始学习率: {config['lr0']}")
    print(f"  余弦学习率: {config['cos_lr']}")
    print(f"  早停patience: {config['patience']}")
    
    # 加载模型
    print(f"\n正在加载模型: {config['model']}")
    model = YOLO(config['model'])
    
    # 数据配置文件路径
    data_yaml = '/workspace/yolo/data_1105/data.yaml'
    
    print(f"\n开始训练...")
    print(f"数据配置: {data_yaml}")
    print(f"结果保存至: {config['project']}/{config['name']}\n")
    
    # 开始训练
    results = model.train(
        # 数据配置
        data=data_yaml,
        
        # 训练参数
        epochs=config['epochs'],
        batch=config['batch'],
        imgsz=config['imgsz'],
        device=config['device'],
        workers=config['workers'],
        
        # 优化器配置
        optimizer=config['optimizer'],
        lr0=config['lr0'],
        lrf=config['lrf'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay'],
        
        # 训练策略
        patience=config['patience'],
        save_period=config['save_period'],
        cos_lr=config['cos_lr'],
        warmup_epochs=config['warmup_epochs'],
        warmup_momentum=config['warmup_momentum'],
        warmup_bias_lr=config['warmup_bias_lr'],
        
        # 数据增强
        hsv_h=config['hsv_h'],
        hsv_s=config['hsv_s'],
        hsv_v=config['hsv_v'],
        degrees=config['degrees'],
        translate=config['translate'],
        scale=config['scale'],
        shear=config['shear'],
        perspective=config['perspective'],
        flipud=config['flipud'],
        fliplr=config['fliplr'],
        mosaic=config['mosaic'],
        mixup=config['mixup'],
        copy_paste=config['copy_paste'],
        
        # 验证和保存
        val=config['val'],
        save=config['save'],
        plots=config['plots'],
        save_json=config['save_json'],
        
        # 输出配置
        project=config['project'],
        name=config['name'],
        exist_ok=config['exist_ok'],
        
        # 其他配置
        pretrained=True,
        verbose=True,
        seed=42,
        deterministic=True,
        amp=True,                    # 自动混合精度训练
    )
    
    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print("=" * 60)
    
    return results


def validate_model(model_path, data_yaml):
    """
    验证训练好的模型
    """
    print("\n" + "=" * 60)
    print("模型验证")
    print("=" * 60)
    
    if not Path(model_path).exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    print(f"正在加载模型: {model_path}")
    model = YOLO(model_path)
    
    print(f"正在验证模型...")
    results = model.val(data=data_yaml)
    
    print(f"\n验证结果:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")
    
    print("=" * 60)
    return results


def main():
    """
    主函数
    """
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "YOLO8s 气泡检测训练" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # 环境检查
    if not check_environment():
        print("\n❌ 环境检查失败，请修复上述问题后重试")
        return
    
    # 训练模型
    results = train_model(CONFIG)
    
    # 获取最佳模型路径
    best_model_path = Path(CONFIG['project']) / CONFIG['name'] / 'weights' / 'best.pt'
    last_model_path = Path(CONFIG['project']) / CONFIG['name'] / 'weights' / 'last.pt'
    
    # 验证最佳模型
    if best_model_path.exists():
        print(f"\n📊 验证最佳模型...")
        validate_model(str(best_model_path), '/workspace/yolo/data_1105/data.yaml')
        
        print(f"\n" + "=" * 60)
        print("训练结果文件:")
        print("=" * 60)
        print(f"  最佳模型: {best_model_path}")
        print(f"  最新模型: {last_model_path}")
        print(f"  训练曲线: {best_model_path.parent.parent}/results.png")
        print(f"  混淆矩阵: {best_model_path.parent.parent}/confusion_matrix.png")
        print("=" * 60)
        
        print(f"\n🚀 使用训练好的模型进行推理:")
        print(f"  python inference.py")
        print()
    else:
        print(f"\n⚠️  未找到最佳模型文件: {best_model_path}")


if __name__ == '__main__':
    main()


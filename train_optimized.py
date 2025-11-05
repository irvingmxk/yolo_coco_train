#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气泡检测最佳训练配置
专门针对 data_1029 数据集优化

数据集特点：
- 训练集：127张图片，997个bubble目标
- 验证集：31张图片
- 平均每张图片：7.8个bubble
- 单类别检测（bubble）

优化策略：
- 小数据集：增强数据增强，避免过拟合
- 高目标密度：使用更大的batch size
- 使用YOLO11s（最新架构，效果更好）
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch

# ============= 最佳配置（针对小数据集优化）=============
CONFIG = {
    # 模型选择：YOLO11s（推荐）vs YOLOv8s
    # YOLO11s: 最新架构，精度更高，速度相当
    # YOLOv8s: 成熟稳定，社区支持好
    'model': 'yolo11s.pt',          # 推荐YOLO11s
    
    # ===== 训练参数（小数据集优化）=====
    'epochs': 200,                   # ✨ 增加到200轮（小数据集需要更多epoch）
    'batch': 16,                     # ✨ 保持16（适合高目标密度）
    'imgsz': 640,                    # 标准尺寸（气泡通常不需要更高分辨率）
    'device': 0,                     # GPU
    'workers': 8,                    # 数据加载线程
    
    # ===== 优化器配置（降低学习率防止过拟合）=====
    'optimizer': 'SGD',              # ✨ 明确使用SGD（小数据集更稳定）
    'lr0': 0.005,                    # ✨ 降低初始学习率（0.01→0.005）
    'lrf': 0.01,                     # 最终学习率因子
    'momentum': 0.937,               # SGD动量
    'weight_decay': 0.001,           # ✨ 增加权重衰减（0.0005→0.001，防止过拟合）
    
    # ===== 训练策略（小数据集需要patience）=====
    'patience': 50,                  # ✨ 大幅增加早停patience（20→50）
    'save_period': 10,               # 每10轮保存
    'cos_lr': True,                  # ✨ 使用余弦退火（更平滑的学习率下降）
    'warmup_epochs': 5,              # ✨ 增加预热轮数（3→5）
    'warmup_momentum': 0.8,          
    'warmup_bias_lr': 0.1,           
    'close_mosaic': 15,              # ✨ 最后15轮关闭mosaic增强
    
    # ===== 数据增强（小数据集需要强增强）=====
    'hsv_h': 0.015,                  # HSV-色调
    'hsv_s': 0.7,                    # HSV-饱和度
    'hsv_v': 0.4,                    # HSV-亮度
    'degrees': 5.0,                  # ✨ 增加旋转（0→5度，聊天截图可能有角度）
    'translate': 0.15,               # ✨ 增加平移（0.1→0.15）
    'scale': 0.6,                    # ✨ 增加缩放（0.5→0.6）
    'shear': 2.0,                    # ✨ 增加剪切（0→2度）
    'perspective': 0.0001,           # ✨ 轻微透视变换
    'flipud': 0.0,                   # 不上下翻转（聊天截图不合理）
    'fliplr': 0.5,                   # 左右翻转（合理）
    'mosaic': 1.0,                   # ✨ Mosaic增强（小数据集必备）
    'mixup': 0.1,                    # ✨ 增加Mixup（0→0.1，增强泛化）
    'copy_paste': 0.1,               # ✨ 增加Copy-paste（0→0.1，增加样本）
    'erasing': 0.4,                  # ✨ 随机擦除（模拟遮挡）
    'crop_fraction': 1.0,            # ✨ 裁剪分数
    
    # ===== 损失函数权重 =====
    'box': 7.5,                      # 边界框损失权重
    'cls': 0.5,                      # 分类损失权重（单类别可降低）
    'dfl': 1.5,                      # 分布式焦点损失
    
    # ===== 验证和保存 =====
    'val': True,                     
    'save': True,                    
    'plots': True,                   
    'save_json': False,              
    
    # ===== 输出配置 =====
    'project': 'runs/train',         
    'name': 'bubble_optimized',      # 新的实验名称
    'exist_ok': True,                
}

# ===== 模型对比分析 =====
MODEL_COMPARISON = """
╔════════════════════════════════════════════════════════════╗
║              YOLO11s vs YOLOv8s 模型对比                   ║
╚════════════════════════════════════════════════════════════╝

📊 性能对比：
┌─────────────┬──────────────┬──────────────┬──────────────┐
│    指标     │   YOLO11s    │   YOLOv8s    │    推荐      │
├─────────────┼──────────────┼──────────────┼──────────────┤
│  精度(mAP)  │    更高      │     高       │  YOLO11s ⭐  │
│  速度(FPS)  │    相当      │    相当      │     相同     │
│  参数量     │    稍少      │    标准      │  YOLO11s ⭐  │
│  架构       │   最新C3k2   │   C2f        │  YOLO11s ⭐  │
│  成熟度     │    较新      │   非常成熟   │   YOLOv8s   │
│  社区支持   │    增长中    │   最广泛     │   YOLOv8s   │
└─────────────┴──────────────┴──────────────┴──────────────┘

🎯 针对你的数据集（127张训练图，单类别bubble）：

✅ 推荐：YOLO11s
理由：
1. ⭐ 精度提升：在小数据集上通常提升2-3% mAP
2. ⭐ 最新架构：C3k2模块，特征提取更强
3. ⭐ 参数更少：过拟合风险更小
4. ⭐ 官方最新：持续优化和支持

⚠️ 备选：YOLOv8s
适用场景：
- 需要最稳定的训练过程
- 需要最多的社区资源
- 项目要求使用成熟技术

💡 建议策略：
1. 先用 YOLO11s 训练（main配置）
2. 如果效果不理想，尝试 YOLOv8s（备用配置）
3. 对比两者在验证集上的表现
"""


def check_environment():
    """检查运行环境"""
    print("=" * 60)
    print("环境检查")
    print("=" * 60)
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA可用")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print(f"⚠️  CUDA不可用，将使用CPU（非常慢）")
        CONFIG['device'] = 'cpu'
    
    # 检查数据集
    data_dir = Path('/workspace/yolo/data_1105')
    train_dir = data_dir / 'train' / 'images'
    val_dir = data_dir / 'val' / 'images'
    
    train_count = len(list(train_dir.glob('*'))) if train_dir.exists() else 0
    val_count = len(list(val_dir.glob('*'))) if val_dir.exists() else 0
    
    print(f"✅ 训练集: {train_count} 张图片")
    print(f"✅ 验证集: {val_count} 张图片")
    print(f"✅ 数据集规模: {'小' if train_count < 500 else '中' if train_count < 2000 else '大'}")
    
    if train_count < 500:
        print(f"\n💡 小数据集建议:")
        print(f"   • 使用强数据增强 ✅ (已启用)")
        print(f"   • 降低学习率 ✅ (lr0=0.005)")
        print(f"   • 增加训练轮数 ✅ (epochs=200)")
        print(f"   • 增加patience ✅ (patience=50)")
        print(f"   • 使用权重衰减 ✅ (weight_decay=0.001)")
    
    print("=" * 60)
    return True


def train_model(config, model_name='YOLO11s'):
    """训练模型"""
    print("\n" + "=" * 60)
    print(f"开始训练 {model_name}")
    print("=" * 60)
    
    # 打印关键配置
    print("\n🔧 关键配置:")
    print(f"  模型: {config['model']}")
    print(f"  训练轮数: {config['epochs']} (小数据集优化)")
    print(f"  批次大小: {config['batch']}")
    print(f"  初始学习率: {config['lr0']} (降低防过拟合)")
    print(f"  权重衰减: {config['weight_decay']} (增加正则化)")
    print(f"  早停patience: {config['patience']} (更有耐心)")
    print(f"  余弦学习率: {config['cos_lr']}")
    print(f"  数据增强: Mosaic + Mixup + Copy-paste ✨")
    
    # 加载模型
    print(f"\n📥 加载模型: {config['model']}")
    model = YOLO(config['model'])
    
    # 数据配置
    data_yaml = '/workspace/yolo/data_1105/data.yaml'
    
    print(f"\n🚀 开始训练...")
    print(f"   数据: {data_yaml}")
    print(f"   输出: {config['project']}/{config['name']}\n")
    
    # 训练
    results = model.train(
        data=data_yaml,
        epochs=config['epochs'],
        batch=config['batch'],
        imgsz=config['imgsz'],
        device=config['device'],
        workers=config['workers'],
        optimizer=config['optimizer'],
        lr0=config['lr0'],
        lrf=config['lrf'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay'],
        patience=config['patience'],
        save_period=config['save_period'],
        cos_lr=config['cos_lr'],
        warmup_epochs=config['warmup_epochs'],
        warmup_momentum=config['warmup_momentum'],
        warmup_bias_lr=config['warmup_bias_lr'],
        close_mosaic=config.get('close_mosaic', 10),
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
        erasing=config.get('erasing', 0.4),
        crop_fraction=config.get('crop_fraction', 1.0),
        box=config.get('box', 7.5),
        cls=config.get('cls', 0.5),
        dfl=config.get('dfl', 1.5),
        val=config['val'],
        save=config['save'],
        plots=config['plots'],
        save_json=config['save_json'],
        project=config['project'],
        name=config['name'],
        exist_ok=config['exist_ok'],
        pretrained=True,
        verbose=True,
        seed=42,
        deterministic=True,
        amp=True,
    )
    
    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print("=" * 60)
    
    return results


def validate_model(model_path, data_yaml):
    """验证模型"""
    print("\n" + "=" * 60)
    print("📊 模型验证")
    print("=" * 60)
    
    if not Path(model_path).exists():
        print(f"❌ 模型不存在: {model_path}")
        return None
    
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    
    print(f"验证中...")
    results = model.val(data=data_yaml, split='val')
    
    print(f"\n📈 验证结果:")
    print(f"  mAP50:     {results.box.map50:.4f}")
    print(f"  mAP50-95:  {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall:    {results.box.mr:.4f}")
    
    # 评估标准
    print(f"\n📊 效果评估:")
    if results.box.map50 > 0.90:
        print(f"  ⭐⭐⭐ 优秀！(mAP50 > 0.90)")
    elif results.box.map50 > 0.80:
        print(f"  ⭐⭐ 良好 (mAP50 > 0.80)")
    elif results.box.map50 > 0.70:
        print(f"  ⭐ 及格 (mAP50 > 0.70)")
    else:
        print(f"  ⚠️  需要改进 (mAP50 < 0.70)")
    
    print("=" * 60)
    return results


def main():
    """主函数"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 12 + "气泡检测最佳训练配置" + " " * 12 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # 显示模型对比
    print(MODEL_COMPARISON)
    
    # 环境检查
    print("\n")
    if not check_environment():
        return
    
    # 确认开始训练
    print("\n" + "=" * 60)
    print("📋 训练配置已优化完成")
    print("=" * 60)
    print(f"✨ 针对小数据集（127张）的优化:")
    print(f"   • 训练轮数: {CONFIG['epochs']}")
    print(f"   • 学习率: {CONFIG['lr0']} (降低)")
    print(f"   • 早停patience: {CONFIG['patience']} (增加)")
    print(f"   • 数据增强: 强化")
    print(f"   • 正则化: weight_decay={CONFIG['weight_decay']}")
    print("=" * 60)
    
    input("\n按 Enter 开始训练（或 Ctrl+C 取消）...")
    
    # 训练
    results = train_model(CONFIG, 'YOLO11s')
    
    # 验证
    best_model = Path(CONFIG['project']) / CONFIG['name'] / 'weights' / 'best.pt'
    if best_model.exists():
        print(f"\n📊 验证最佳模型...")
        validate_model(str(best_model), '/workspace/yolo/data_1105/data.yaml')
        
        print(f"\n" + "=" * 60)
        print("📁 训练结果:")
        print("=" * 60)
        print(f"  最佳模型: {best_model}")
        print(f"  训练曲线: {best_model.parent.parent}/results.png")
        print(f"  混淆矩阵: {best_model.parent.parent}/confusion_matrix.png")
        print(f"  PR曲线:   {best_model.parent.parent}/PR_curve.png")
        print("=" * 60)
        
        print(f"\n🎯 下一步:")
        print(f"  1. 查看训练曲线，确认是否收敛")
        print(f"  2. 查看混淆矩阵，分析错误类型")
        print(f"  3. 使用 bubble_ocr.py 进行实际测试")
        print(f"  4. 如果效果不理想，尝试:")
        print(f"     - 增加数据集")
        print(f"     - 调整增强策略")
        print(f"     - 尝试 YOLOv8s")
        print()


if __name__ == '__main__':
    main()


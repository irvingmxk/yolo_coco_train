#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备 data_1029 数据集
将数据划分为训练集和验证集，并生成配置文件
"""

import os
import shutil
from pathlib import Path
import random
import yaml

def split_dataset(data_dir, val_split=0.2, seed=42):
    """
    将数据集分割为训练集和验证集
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / 'images'
    labels_dir = data_dir / 'labels'
    
    print("=" * 60)
    print("准备数据集: data_1105")
    print("=" * 60)
    
    # 获取所有图片文件
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    image_files = sorted(image_files)
    
    print(f"\n✅ 找到 {len(image_files)} 张图片")
    
    # 随机打乱
    random.seed(seed)
    random.shuffle(image_files)
    
    # 分割
    val_size = int(len(image_files) * val_split)
    train_files = image_files[val_size:]
    val_files = image_files[:val_size]
    
    print(f"📊 数据划分:")
    print(f"   训练集: {len(train_files)} 张 ({(1-val_split)*100:.0f}%)")
    print(f"   验证集: {len(val_files)} 张 ({val_split*100:.0f}%)")
    
    # 创建目录结构
    for split in ['train', 'val']:
        (data_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (data_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 复制文件到对应目录
    def copy_files(file_list, split_name):
        copied = 0
        for img_path in file_list:
            label_path = labels_dir / (img_path.stem + '.txt')
            
            # 复制图片
            dst_img = data_dir / split_name / 'images' / img_path.name
            shutil.copy(img_path, dst_img)
            
            # 复制标签（如果存在）
            if label_path.exists():
                dst_label = data_dir / split_name / 'labels' / label_path.name
                shutil.copy(label_path, dst_label)
                copied += 1
        return copied
    
    print(f"\n📁 正在复制文件...")
    train_copied = copy_files(train_files, 'train')
    val_copied = copy_files(val_files, 'val')
    
    print(f"   训练集: {train_copied}/{len(train_files)} 个标签文件")
    print(f"   验证集: {val_copied}/{len(val_files)} 个标签文件")
    
    if train_copied < len(train_files) or val_copied < len(val_files):
        print(f"\n⚠️  部分图片没有对应的标签文件")
    
    print(f"\n✅ 数据集分割完成！")
    return len(train_files), len(val_files)

def create_data_yaml(data_dir):
    """
    创建YOLO数据配置文件
    """
    data_dir = Path(data_dir).absolute()
    classes_file = data_dir / 'classes.txt'
    
    # 读取类别
    with open(classes_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f if line.strip()]
    
    print(f"\n📋 检测类别:")
    for i, name in enumerate(class_names):
        print(f"   {i}: {name}")
    
    # 创建配置
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
    
    print(f"\n✅ 配置文件已创建: {yaml_path}")
    return yaml_path

def check_data_quality(data_dir):
    """
    检查数据质量
    """
    data_dir = Path(data_dir)
    
    print(f"\n" + "=" * 60)
    print("数据质量检查")
    print("=" * 60)
    
    for split in ['train', 'val']:
        images_dir = data_dir / split / 'images'
        labels_dir = data_dir / split / 'labels'
        
        images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        labels = list(labels_dir.glob('*.txt'))
        
        print(f"\n{split.upper()}:")
        print(f"  图片数量: {len(images)}")
        print(f"  标签数量: {len(labels)}")
        
        if len(images) != len(labels):
            print(f"  ⚠️  图片和标签数量不匹配")
        else:
            print(f"  ✅ 图片和标签数量匹配")
        
        # 检查空标签
        empty_labels = 0
        for label_file in labels:
            if label_file.stat().st_size == 0:
                empty_labels += 1
        
        if empty_labels > 0:
            print(f"  ⚠️  有 {empty_labels} 个空标签文件")
        else:
            print(f"  ✅ 无空标签文件")

def main():
    """
    主函数
    """
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "准备 data_1105 数据集" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # 数据目录
    data_dir = Path('/workspace/yolo/data_1105')
    
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    # 检查是否已经分割
    if (data_dir / 'train').exists() and (data_dir / 'val').exists():
        print("⚠️  数据集已经分割过了")
        response = input("是否重新分割? (y/n): ").lower()
        if response != 'y':
            print("跳过数据分割")
        else:
            # 删除旧的分割
            print("正在删除旧的分割...")
            shutil.rmtree(data_dir / 'train', ignore_errors=True)
            shutil.rmtree(data_dir / 'val', ignore_errors=True)
            
            # 重新分割
            split_dataset(data_dir, val_split=0.2)
    else:
        # 分割数据集
        split_dataset(data_dir, val_split=0.2)
    
    # 创建配置文件
    yaml_path = create_data_yaml(data_dir)
    
    # 检查数据质量
    check_data_quality(data_dir)
    
    print("\n" + "=" * 60)
    print("数据准备完成！")
    print("=" * 60)
    print(f"\n📂 数据目录: {data_dir}")
    print(f"📄 配置文件: {yaml_path}")
    print(f"\n🚀 现在可以开始训练了:")
    print(f"   python train_yolo11n.py")
    print()

if __name__ == '__main__':
    main()
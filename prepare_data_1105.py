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

def split_dataset(data_dir, val_split=0.2, seed=42, filter_class=None):
    """
    将数据集分割为训练集和验证集
    
    Args:
        data_dir: 数据目录
        val_split: 验证集比例
        seed: 随机种子
        filter_class: 要保留的类别ID（None表示保留所有，0表示只保留bubble）
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
    def copy_files(file_list, split_name, filter_class=None):
        """
        复制文件并过滤标签
        
        Args:
            file_list: 文件列表
            split_name: 分割名称（train/val）
            filter_class: 要保留的类别ID（None表示保留所有，0表示只保留bubble）
        """
        copied = 0
        filtered_count = 0
        for img_path in file_list:
            label_path = labels_dir / (img_path.stem + '.txt')
            
            # 复制图片
            dst_img = data_dir / split_name / 'images' / img_path.name
            shutil.copy(img_path, dst_img)
            
            # 复制标签（如果存在）
            if label_path.exists():
                dst_label = data_dir / split_name / 'labels' / label_path.name
                
                # 如果需要过滤类别
                if filter_class is not None:
                    # 读取标签文件
                    with open(label_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # 过滤：只保留指定类别的标注
                    filtered_lines = []
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if class_id == filter_class:
                                # 类别ID改为0（因为现在只有一个类别）
                                filtered_lines.append(f"0 {' '.join(parts[1:])}\n")
                    
                    # 写入过滤后的标签
                    with open(dst_label, 'w', encoding='utf-8') as f:
                        f.writelines(filtered_lines)
                    
                    # 统计过滤掉的标注
                    original_count = len([l for l in lines if l.strip()])
                    filtered_count += original_count - len(filtered_lines)
                    
                    if len(filtered_lines) > 0:
                        copied += 1
                else:
                    # 不过滤，直接复制
                    shutil.copy(label_path, dst_label)
                    copied += 1
        return copied, filtered_count
    
    print(f"\n📁 正在复制文件...")
    if filter_class is not None:
        print(f"   过滤类别: 只保留类别 {filter_class} (bubble)")
    
    train_copied, train_filtered = copy_files(train_files, 'train', filter_class)
    val_copied, val_filtered = copy_files(val_files, 'val', filter_class)
    
    print(f"   训练集: {train_copied}/{len(train_files)} 个标签文件")
    print(f"   验证集: {val_copied}/{len(val_files)} 个标签文件")
    
    if filter_class is not None:
        print(f"   过滤统计: 训练集过滤 {train_filtered} 个标注，验证集过滤 {val_filtered} 个标注")
    
    if train_copied < len(train_files) or val_copied < len(val_files):
        print(f"\n⚠️  部分图片没有对应的标签文件")
    
    print(f"\n✅ 数据集分割完成！")
    return len(train_files), len(val_files)

def create_data_yaml(data_dir, filter_class=None):
    """
    创建YOLO数据配置文件
    
    Args:
        data_dir: 数据目录
        filter_class: 要保留的类别ID（None表示保留所有，0表示只保留bubble）
    """
    data_dir = Path(data_dir).absolute()
    classes_file = data_dir / 'classes.txt'
    
    # 读取类别
    with open(classes_file, 'r', encoding='utf-8') as f:
        all_class_names = [line.strip() for line in f if line.strip()]
    
    # 如果过滤类别，只使用指定的类别
    if filter_class is not None:
        class_names = [all_class_names[filter_class]]
        print(f"\n📋 检测类别（已过滤）:")
        print(f"   0: {class_names[0]} (原类别 {filter_class})")
    else:
        class_names = all_class_names
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
    
    # 配置：只训练bubble标签
    FILTER_CLASS = 0  # 0 = bubble, 1 = keyboard, None = 保留所有
    print(f"\n🎯 训练配置: 只保留类别 {FILTER_CLASS} (bubble)")
    print(f"   将过滤掉 keyboard 标注")
    
    # 检查是否已经分割
    if (data_dir / 'train').exists() and (data_dir / 'val').exists():
        print("\n⚠️  数据集已经分割过了")
        response = input("是否重新分割? (y/n): ").lower()
        if response != 'y':
            print("跳过数据分割")
        else:
            # 删除旧的分割
            print("正在删除旧的分割...")
            shutil.rmtree(data_dir / 'train', ignore_errors=True)
            shutil.rmtree(data_dir / 'val', ignore_errors=True)
            
            # 重新分割（过滤类别）
            split_dataset(data_dir, val_split=0.2, filter_class=FILTER_CLASS)
    else:
        # 分割数据集（过滤类别）
        split_dataset(data_dir, val_split=0.2, filter_class=FILTER_CLASS)
    
    # 创建配置文件（过滤类别）
    yaml_path = create_data_yaml(data_dir, filter_class=FILTER_CLASS)
    
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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR 系统环境测试脚本
检查所有依赖是否正确安装
"""

import sys
from pathlib import Path

print("╔════════════════════════════════════════════════════════════╗")
print("║          OCR 系统环境检查                                   ║")
print("╚════════════════════════════════════════════════════════════╝")
print()

# 检查基础依赖
print("1. 检查基础依赖...")
print("-" * 60)

try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"❌ OpenCV 未安装: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy 未安装: {e}")
    sys.exit(1)

try:
    from ultralytics import YOLO
    print(f"✅ Ultralytics YOLO")
except ImportError as e:
    print(f"❌ Ultralytics 未安装: {e}")
    sys.exit(1)

# 检查 OCR 引擎
print()
print("2. 检查 OCR 引擎...")
print("-" * 60)

# Tesseract
try:
    import pytesseract
    version = pytesseract.get_tesseract_version()
    print(f"✅ Tesseract-OCR: {version}")
    
    # 检查支持的语言
    try:
        import subprocess
        result = subprocess.run(['tesseract', '--list-langs'], 
                              capture_output=True, text=True)
        langs = result.stdout.split('\n')[1:]  # 跳过第一行
        langs = [l.strip() for l in langs if l.strip()]
        
        has_chinese = any('chi_sim' in l for l in langs)
        has_english = any('eng' in l for l in langs)
        
        if has_chinese and has_english:
            print(f"   支持语言: 中文 ✅, 英文 ✅")
        elif has_english:
            print(f"   支持语言: 英文 ✅, 中文 ❌")
            print(f"   提示: 安装中文语言包以支持中文识别")
            print(f"   命令: sudo apt-get install tesseract-ocr-chi-sim")
        else:
            print(f"   支持语言: 未检测到常用语言")
    except:
        pass
        
except ImportError:
    print("❌ pytesseract 未安装")
    print("   安装命令: pip install pytesseract")
except Exception as e:
    print(f"❌ Tesseract-OCR 未安装或配置错误: {e}")
    print("   安装命令: bash install_tesseract.sh")

# PaddleOCR (可选)
try:
    from paddleocr import PaddleOCR
    print("✅ PaddleOCR (可选)")
except ImportError:
    print("⚪ PaddleOCR 未安装 (可选，不影响使用)")

# EasyOCR (可选)
try:
    import easyocr
    print("✅ EasyOCR (可选)")
except ImportError:
    print("⚪ EasyOCR 未安装 (可选，不影响使用)")

# 检查 YOLO 模型
print()
print("3. 检查 YOLO 模型...")
print("-" * 60)

model_paths = [
    'runs/train/yolo11n_bubble/weights/best.pt',
    'runs/train/bubble_detection/weights/best.pt',
    'yolo11n.pt'
]

found_model = False
for model_path in model_paths:
    if Path(model_path).exists():
        print(f"✅ 找到模型: {model_path}")
        found_model = True
        break

if not found_model:
    print("❌ 未找到训练好的 YOLO 模型")
    print("   请先训练模型: python train_yolo11n.py")

# 检查测试数据
print()
print("4. 检查测试数据...")
print("-" * 60)

test_data_paths = [
    'data/val/images',
    'data/train/images'
]

found_data = False
for data_path in test_data_paths:
    data_dir = Path(data_path)
    if data_dir.exists():
        images = list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.png'))
        if images:
            print(f"✅ 找到测试数据: {data_path} ({len(images)} 张图片)")
            found_data = True
            break

if not found_data:
    print("⚠️  未找到测试图片")

# 检查核心模块
print()
print("5. 检查核心模块...")
print("-" * 60)

try:
    from bubble_ocr import (
        load_yolo_model,
        detect_bubbles,
        crop_bubble,
        init_ocr_engine,
        recognize_text,
        process_image
    )
    print("✅ bubble_ocr 模块导入成功")
    print("   - load_yolo_model ✅")
    print("   - detect_bubbles ✅")
    print("   - crop_bubble ✅")
    print("   - init_ocr_engine ✅")
    print("   - recognize_text ✅")
    print("   - process_image ✅")
except Exception as e:
    print(f"❌ bubble_ocr 模块导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 总结
print()
print("╔════════════════════════════════════════════════════════════╗")
print("║          检查完成                                           ║")
print("╚════════════════════════════════════════════════════════════╝")
print()

if found_model and found_data:
    print("✅ 系统就绪！可以开始使用")
    print()
    print("快速开始:")
    print("  # 测试单张图片")
    print("  python run_ocr.py --image data/val/images/图片名.jpg")
    print()
    print("  # 批量处理")
    print("  python run_ocr.py --dir data/val/images")
    print()
elif not found_model:
    print("⚠️  请先训练 YOLO 模型:")
    print("  python train_yolo11n.py")
    print()
else:
    print("⚠️  系统可用，但缺少测试数据")
    print()


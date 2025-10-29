#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气泡检测 + OCR 识别 - 命令行工具
简化版接口，方便快速使用
"""

import argparse
import sys
from pathlib import Path
from bubble_ocr import (
    load_yolo_model,
    init_ocr_engine,
    process_image,
    CONFIG
)


def process_single_image(image_path: str, yolo_model_path: str = None, 
                        ocr_backend: str = 'tesseract', ocr_lang: str = 'chi_sim+eng',
                        conf: float = 0.25):
    """
    处理单张图片
    """
    # 更新配置
    if yolo_model_path:
        CONFIG['yolo_model'] = yolo_model_path
    CONFIG['ocr_backend'] = ocr_backend
    CONFIG['ocr_lang'] = ocr_lang
    CONFIG['yolo_conf'] = conf
    
    # 加载模型
    print("正在加载模型...")
    yolo_model = load_yolo_model(CONFIG['yolo_model'])
    ocr_engine = init_ocr_engine(backend=CONFIG['ocr_backend'], lang=CONFIG['ocr_lang'])
    
    # 处理图片
    result = process_image(image_path, yolo_model, ocr_engine, CONFIG)
    
    # 打印详细结果
    print("\n" + "="*60)
    print("识别结果汇总:")
    print("="*60)
    
    for i, (det, ocr) in enumerate(zip(result['detections'], result['ocr_results'])):
        print(f"\n气泡 #{i+1}:")
        print(f"  位置: {det['bbox']}")
        print(f"  检测置信度: {det['confidence']:.2f}")
        if ocr['text']:
            print(f"  识别文字:")
            for line in ocr['text'].split('\n'):
                print(f"    {line}")
            print(f"  OCR置信度: {ocr['confidence']:.2f}")
        else:
            print(f"  识别文字: (无)")
    
    return result


def process_directory(dir_path: str, yolo_model_path: str = None,
                     ocr_backend: str = 'tesseract', ocr_lang: str = 'chi_sim+eng',
                     conf: float = 0.25):
    """
    批量处理目录中的图片
    """
    # 更新配置
    if yolo_model_path:
        CONFIG['yolo_model'] = yolo_model_path
    CONFIG['ocr_backend'] = ocr_backend
    CONFIG['ocr_lang'] = ocr_lang
    CONFIG['yolo_conf'] = conf
    
    # 加载模型（只加载一次）
    print("正在加载模型...")
    yolo_model = load_yolo_model(CONFIG['yolo_model'])
    ocr_engine = init_ocr_engine(backend=CONFIG['ocr_backend'], lang=CONFIG['ocr_lang'])
    
    # 获取所有图片
    dir_path = Path(dir_path)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(dir_path.glob(f'*{ext}'))
        image_files.extend(dir_path.glob(f'*{ext.upper()}'))
    
    print(f"\n找到 {len(image_files)} 张图片")
    
    # 批量处理
    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 处理: {image_path.name}")
        try:
            result = process_image(str(image_path), yolo_model, ocr_engine, CONFIG)
            results.append(result)
        except Exception as e:
            print(f"❌ 处理失败: {e}")
    
    # 统计
    total_bubbles = sum(len(r['detections']) for r in results)
    total_texts = sum(sum(1 for ocr in r['ocr_results'] if ocr['text']) for r in results)
    
    print(f"\n" + "="*60)
    print(f"批量处理完成:")
    print(f"  处理图片: {len(results)}/{len(image_files)}")
    print(f"  检测气泡: {total_bubbles} 个")
    print(f"  识别文字: {total_texts} 个")
    print("="*60)
    
    return results


def main():
    """
    命令行入口
    """
    parser = argparse.ArgumentParser(
        description='气泡检测 + OCR 文字识别工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理单张图片（默认使用Tesseract）
  python run_ocr.py --image test.jpg
  
  # 批量处理目录
  python run_ocr.py --dir data/val/images
  
  # 指定YOLO模型
  python run_ocr.py --image test.jpg --model runs/train/yolo11n_bubble/weights/best.pt
  
  # 只识别英文
  python run_ocr.py --image test.jpg --lang eng
  
  # 使用PaddleOCR（需要先安装）
  python run_ocr.py --image test.jpg --ocr paddleocr --lang ch
  
  # 调整检测置信度
  python run_ocr.py --image test.jpg --conf 0.5
        """
    )
    
    # 输入参数
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', '-i', type=str, help='输入图片路径')
    input_group.add_argument('--dir', '-d', type=str, help='输入目录路径')
    
    # 模型参数
    parser.add_argument('--model', '-m', type=str, 
                       default='runs/train/yolo11n_bubble/weights/best.pt',
                       help='YOLO模型路径 (默认: runs/train/yolo11n_bubble/weights/best.pt)')
    
    # OCR参数
    parser.add_argument('--ocr', type=str, default='tesseract',
                       choices=['tesseract', 'paddleocr', 'easyocr'],
                       help='OCR引擎 (默认: tesseract)')
    parser.add_argument('--lang', type=str, default='chi_sim+eng',
                       help='OCR语言 (默认: chi_sim+eng=中英文, eng=英文, chi_sim=简体中文)')
    
    # 检测参数
    parser.add_argument('--conf', type=float, default=0.25,
                       help='YOLO检测置信度阈值 (默认: 0.25)')
    
    args = parser.parse_args()
    
    try:
        if args.image:
            # 处理单张图片
            if not Path(args.image).exists():
                print(f"❌ 错误: 图片不存在: {args.image}")
                sys.exit(1)
            
            process_single_image(
                args.image,
                yolo_model_path=args.model,
                ocr_backend=args.ocr,
                ocr_lang=args.lang,
                conf=args.conf
            )
            
        elif args.dir:
            # 批量处理目录
            if not Path(args.dir).exists():
                print(f"❌ 错误: 目录不存在: {args.dir}")
                sys.exit(1)
            
            process_directory(
                args.dir,
                yolo_model_path=args.model,
                ocr_backend=args.ocr,
                ocr_lang=args.lang,
                conf=args.conf
            )
    
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


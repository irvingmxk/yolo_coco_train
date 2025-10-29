#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气泡检测 + OCR 文字识别系统
使用 YOLO11n 检测气泡框，然后使用 OCR 识别框内文字
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============= 配置参数 =============
CONFIG = {
    # YOLO模型配置
    # 'yolo_model': 'runs/train/yolo11n_bubble/weights/best.pt',  # YOLO模型路径
    'yolo_model': 'runs/train/bubble_detection/weights/best.pt',  # YOLO8s模型路径

    'yolo_conf': 0.25,        # YOLO置信度阈值
    'yolo_iou': 0.45,         # YOLO NMS IOU阈值
    
    # OCR配置
    'ocr_backend': 'tesseract',  # 'tesseract', 'paddleocr' 或 'easyocr'
    'ocr_lang': 'chi_sim+eng',   # Tesseract语言：'chi_sim+eng'=中英文, 'eng'=英文
    'use_multithread': True,     # 是否使用多线程（仅Tesseract）
    'num_workers': 12,           # 线程数（96核服务器优化为12）
    
    # 图像处理
    'padding': 5,                # 裁剪框的边距
    'min_text_confidence': 0.5,  # OCR最小置信度
    
    # 输出配置
    'save_crops': True,          # 是否保存裁剪的气泡图片
    'save_annotated': True,      # 是否保存标注图片
    'output_dir': 'runs/ocr',    # 输出目录
}


# ============= 1. YOLO 检测模块 =============

def load_yolo_model(model_path: str) -> YOLO:
    """
    加载 YOLO 模型
    
    Args:
        model_path: 模型权重路径
        
    Returns:
        YOLO模型对象
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"加载 YOLO 模型: {model_path}")
    model = YOLO(model_path)
    return model


def detect_bubbles(model: YOLO, image_path: str, conf: float = 0.25, iou: float = 0.45) -> List[Dict]:
    """
    使用 YOLO 检测图片中的气泡
    
    Args:
        model: YOLO模型
        image_path: 图片路径
        conf: 置信度阈值
        iou: NMS IOU阈值
        
    Returns:
        检测结果列表，每个元素包含：
        {
            'bbox': [x1, y1, x2, y2],  # 边界框坐标
            'confidence': float,        # 置信度
            'class_id': int,            # 类别ID
            'class_name': str           # 类别名称
        }
    """
    print(f"\n检测气泡: {image_path}")
    
    # 进行检测
    results = model.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        verbose=False
    )
    
    # 解析结果
    detections = []
    if len(results) > 0:
        result = results[0]
        boxes = result.boxes
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = result.names[class_id]
            
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': confidence,
                'class_id': class_id,
                'class_name': class_name
            })
    
    print(f"检测到 {len(detections)} 个气泡")
    return detections


# ============= 2. 图像预处理模块 =============

def crop_bubble(image: np.ndarray, bbox: List[int], padding: int = 5) -> np.ndarray:
    """
    从原图裁剪气泡区域
    
    Args:
        image: 原始图片（numpy数组）
        bbox: 边界框 [x1, y1, x2, y2]
        padding: 边距（像素）
        
    Returns:
        裁剪后的图片
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    
    # 添加边距，但不超出图片范围
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    # 裁剪
    cropped = image[y1:y2, x1:x2]
    return cropped


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    图像预处理，提高OCR识别率
    
    Args:
        image: 输入图片
        
    Returns:
        预处理后的图片
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 去噪
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # 自适应二值化（对不同光照条件更鲁棒）
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    
    # 轻微膨胀，连接断裂的文字
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.dilate(binary, kernel, iterations=1)
    
    return processed


# ============= 3. OCR 识别模块 =============

def recognize_text_tesseract(ocr_config: Dict, image: np.ndarray, min_confidence: float = 0.5) -> Dict:
    """
    使用 Tesseract-OCR 识别文字
    
    Args:
        ocr_config: Tesseract配置字典
        image: 输入图片
        min_confidence: 最小置信度
        
    Returns:
        识别结果字典：
        {
            'text': str,           # 完整文本
            'lines': List[Dict],   # 每行文本详情
            'confidence': float    # 平均置信度
        }
    """
    pytesseract = ocr_config['pytesseract']
    lang = ocr_config['lang']
    
    # 获取详细的OCR数据
    try:
        # 使用 image_to_data 获取详细信息（包括置信度）
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
        
        # 提取文本和置信度
        lines = []
        current_line = []
        current_confidences = []
        last_line_num = -1
        
        for i in range(len(data['text'])):
            conf = int(data['conf'][i])
            text = data['text'][i].strip()
            line_num = data['line_num'][i]
            
            # 过滤空文本和低置信度
            if text and conf > 0:
                if line_num != last_line_num and current_line:
                    # 新的一行，保存上一行
                    line_text = ' '.join(current_line)
                    avg_conf = np.mean(current_confidences) / 100.0
                    
                    if avg_conf >= min_confidence:
                        lines.append({
                            'text': line_text,
                            'confidence': avg_conf,
                            'bbox': None  # Tesseract的bbox格式不同，这里简化
                        })
                    
                    current_line = []
                    current_confidences = []
                
                current_line.append(text)
                current_confidences.append(conf)
                last_line_num = line_num
        
        # 处理最后一行
        if current_line:
            line_text = ' '.join(current_line)
            avg_conf = np.mean(current_confidences) / 100.0
            if avg_conf >= min_confidence:
                lines.append({
                    'text': line_text,
                    'confidence': avg_conf,
                    'bbox': None
                })
        
        # 合并所有行
        full_text = '\n'.join([line['text'] for line in lines])
        avg_confidence = np.mean([line['confidence'] for line in lines]) if lines else 0.0
        
        return {
            'text': full_text,
            'lines': lines,
            'confidence': float(avg_confidence)
        }
        
    except Exception as e:
        print(f"  ⚠️  OCR识别出错: {e}")
        return {
            'text': '',
            'lines': [],
            'confidence': 0.0
        }


def init_ocr_engine(backend: str = 'tesseract', lang: str = 'chi_sim+eng'):
    """
    初始化 OCR 引擎
    
    Args:
        backend: OCR后端 ('tesseract', 'paddleocr' 或 'easyocr')
        lang: 语言代码
        
    Returns:
        OCR引擎对象或配置字典
    """
    print(f"\n初始化 OCR 引擎: {backend} (语言: {lang})")
    
    if backend == 'tesseract':
        try:
            import pytesseract
            # 测试是否可用
            pytesseract.get_tesseract_version()
            print("✅ Tesseract-OCR 初始化成功")
            # 返回配置字典
            return {
                'engine': 'tesseract',
                'lang': lang,
                'pytesseract': pytesseract
            }
        except ImportError:
            print("❌ pytesseract 未安装")
            print("安装命令: pip install pytesseract")
            raise
        except Exception as e:
            print(f"❌ Tesseract-OCR 未安装或配置错误: {e}")
            print("\n安装 Tesseract-OCR:")
            print("  Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim")
            print("  CentOS/RHEL: sudo yum install tesseract tesseract-langpack-chi-sim")
            print("  macOS: brew install tesseract tesseract-lang")
            raise
            
    elif backend == 'paddleocr':
        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(
                use_angle_cls=True,  # 使用方向分类器
                lang=lang,
                show_log=False,
                use_gpu=True
            )
            print("✅ PaddleOCR 初始化成功")
            return ocr
        except ImportError:
            print("❌ PaddleOCR 未安装")
            print("安装命令: pip install paddleocr paddlepaddle-gpu")
            raise
            
    elif backend == 'easyocr':
        try:
            import easyocr
            lang_map = {'ch': ['ch_sim', 'en'], 'en': ['en']}
            langs = lang_map.get(lang, ['en'])
            ocr = easyocr.Reader(langs, gpu=True)
            print("✅ EasyOCR 初始化成功")
            return ocr
        except ImportError:
            print("❌ EasyOCR 未安装")
            print("安装命令: pip install easyocr")
            raise
    else:
        raise ValueError(f"不支持的OCR后端: {backend}")


def recognize_text_paddleocr(ocr_engine, image: np.ndarray, min_confidence: float = 0.5) -> Dict:
    """
    使用 PaddleOCR 识别文字
    
    Args:
        ocr_engine: PaddleOCR对象
        image: 输入图片
        min_confidence: 最小置信度
        
    Returns:
        识别结果字典：
        {
            'text': str,           # 完整文本
            'lines': List[Dict],   # 每行文本详情
            'confidence': float    # 平均置信度
        }
    """
    result = ocr_engine.ocr(image, cls=True)
    
    if result is None or len(result) == 0 or result[0] is None:
        return {
            'text': '',
            'lines': [],
            'confidence': 0.0
        }
    
    lines = []
    text_parts = []
    confidences = []
    
    for line in result[0]:
        box = line[0]  # 文本框坐标
        text_info = line[1]  # (文本, 置信度)
        text = text_info[0]
        confidence = text_info[1]
        
        if confidence >= min_confidence:
            lines.append({
                'text': text,
                'confidence': confidence,
                'bbox': box
            })
            text_parts.append(text)
            confidences.append(confidence)
    
    # 合并所有行
    full_text = '\n'.join(text_parts)
    avg_confidence = np.mean(confidences) if confidences else 0.0
    
    return {
        'text': full_text,
        'lines': lines,
        'confidence': float(avg_confidence)
    }


def recognize_text_easyocr(ocr_engine, image: np.ndarray, min_confidence: float = 0.5) -> Dict:
    """
    使用 EasyOCR 识别文字
    
    Args:
        ocr_engine: EasyOCR Reader对象
        image: 输入图片
        min_confidence: 最小置信度
        
    Returns:
        识别结果字典（格式同 PaddleOCR）
    """
    results = ocr_engine.readtext(image)
    
    lines = []
    text_parts = []
    confidences = []
    
    for detection in results:
        box = detection[0]  # 文本框坐标
        text = detection[1]  # 文本
        confidence = detection[2]  # 置信度
        
        if confidence >= min_confidence:
            lines.append({
                'text': text,
                'confidence': confidence,
                'bbox': box
            })
            text_parts.append(text)
            confidences.append(confidence)
    
    full_text = '\n'.join(text_parts)
    avg_confidence = np.mean(confidences) if confidences else 0.0
    
    return {
        'text': full_text,
        'lines': lines,
        'confidence': float(avg_confidence)
    }


def recognize_text(ocr_engine, image: np.ndarray, backend: str, 
                  min_confidence: float = 0.5, preprocess: bool = True) -> Dict:
    """
    通用 OCR 文字识别接口（单张图片）
    
    Args:
        ocr_engine: OCR引擎对象或配置字典
        image: 输入图片
        backend: OCR后端类型
        min_confidence: 最小置信度
        preprocess: 是否预处理
        
    Returns:
        识别结果字典
    """
    # 根据后端选择识别函数和预处理策略
    if backend == 'tesseract':
        # Tesseract 使用原图效果更好（已经在函数内部处理）
        return recognize_text_tesseract(ocr_engine, image, min_confidence)
    
    # 对其他OCR引擎进行预处理
    if preprocess:
        processed_image = preprocess_for_ocr(image)
    else:
        processed_image = image
    
    if backend == 'paddleocr':
        return recognize_text_paddleocr(ocr_engine, processed_image, min_confidence)
    elif backend == 'easyocr':
        return recognize_text_easyocr(ocr_engine, processed_image, min_confidence)
    else:
        raise ValueError(f"不支持的OCR后端: {backend}")


def recognize_text_batch(ocr_engine, images: List[np.ndarray], backend: str,
                        min_confidence: float = 0.5, preprocess: bool = True,
                        use_multithread: bool = False, num_workers: int = 4) -> List[Dict]:
    """
    批量 OCR 文字识别接口
    
    Args:
        ocr_engine: OCR引擎对象或配置字典
        images: 输入图片列表
        backend: OCR后端类型
        min_confidence: 最小置信度
        preprocess: 是否预处理
        use_multithread: 是否使用多线程（仅Tesseract）
        num_workers: 线程数
        
    Returns:
        识别结果字典列表
    """
    if len(images) == 0:
        return []
    
    # Tesseract 支持多线程并行处理
    if backend == 'tesseract':
        if use_multithread and len(images) > 1:
            # 多线程并行处理
            results = [None] * len(images)  # 预分配结果列表
            
            def process_single(idx_img):
                idx, img = idx_img
                return idx, recognize_text_tesseract(ocr_engine, img, min_confidence)
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # 提交所有任务
                futures = {executor.submit(process_single, (i, img)): i 
                          for i, img in enumerate(images)}
                
                # 收集结果
                for future in as_completed(futures):
                    idx, result = future.result()
                    results[idx] = result
            
            return results
        else:
            # 单线程处理
            results = []
            for img in images:
                result = recognize_text_tesseract(ocr_engine, img, min_confidence)
                results.append(result)
            return results
    
    # 对于其他引擎，逐张处理
    elif backend in ['paddleocr', 'easyocr']:
        results = []
        for img in images:
            result = recognize_text(ocr_engine, img, backend, min_confidence, preprocess)
            results.append(result)
        return results
    
    else:
        raise ValueError(f"不支持的OCR后端: {backend}")


# ============= 4. 结果可视化模块 =============

def draw_results(image: np.ndarray, detections: List[Dict], 
                ocr_results: List[Dict]) -> np.ndarray:
    """
    在图片上绘制检测框和识别文字
    
    Args:
        image: 原始图片
        detections: YOLO检测结果
        ocr_results: OCR识别结果
        
    Returns:
        标注后的图片
    """
    annotated = image.copy()
    
    for i, (detection, ocr_result) in enumerate(zip(detections, ocr_results)):
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        confidence = detection['confidence']
        text = ocr_result['text']
        ocr_conf = ocr_result['confidence']
        
        # 绘制边界框
        color = (0, 255, 0)  # 绿色
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # 准备标签
        label_lines = []
        label_lines.append(f"#{i+1} Conf:{confidence:.2f}")
        
        # 添加识别的文字（如果有）
        if text:
            # 将多行文本分割显示
            text_lines = text.split('\n')
            for line in text_lines[:3]:  # 最多显示3行
                if len(line) > 20:
                    line = line[:20] + '...'
                label_lines.append(line)
            label_lines.append(f"OCR:{ocr_conf:.2f}")
        else:
            label_lines.append("无文字")
        
        # 绘制标签背景和文字
        y_offset = y1 - 10
        for line in label_lines:
            # 计算文字大小
            (text_width, text_height), baseline = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # 绘制背景
            cv2.rectangle(
                annotated,
                (x1, y_offset - text_height - 5),
                (x1 + text_width + 5, y_offset + baseline),
                color, -1
            )
            
            # 绘制文字
            cv2.putText(
                annotated, line,
                (x1 + 2, y_offset - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA
            )
            
            y_offset -= (text_height + baseline + 5)
    
    return annotated


# ============= 5. 结果保存模块 =============

def save_results(image_path: str, detections: List[Dict], 
                ocr_results: List[Dict], config: Dict) -> Dict:
    """
    保存检测和识别结果
    
    Args:
        image_path: 原始图片路径
        detections: YOLO检测结果
        ocr_results: OCR识别结果
        config: 配置参数
        
    Returns:
        保存路径信息
    """
    # 创建输出目录
    output_dir = Path(config['output_dir'])
    image_name = Path(image_path).stem
    
    result_dir = output_dir / image_name
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取原图
    image = cv2.imread(image_path)
    
    saved_files = {
        'result_dir': str(result_dir),
        'crops': [],
        'json': None,
        'annotated': None
    }
    
    # 保存裁剪的气泡图片
    if config['save_crops']:
        crops_dir = result_dir / 'crops'
        crops_dir.mkdir(exist_ok=True)
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            cropped = crop_bubble(image, bbox, config['padding'])
            
            crop_path = crops_dir / f'bubble_{i+1}.jpg'
            cv2.imwrite(str(crop_path), cropped)
            saved_files['crops'].append(str(crop_path))
    
    # 保存标注图片
    if config['save_annotated']:
        annotated = draw_results(image, detections, ocr_results)
        annotated_path = result_dir / 'annotated.jpg'
        cv2.imwrite(str(annotated_path), annotated)
        saved_files['annotated'] = str(annotated_path)
    
    # 保存JSON结果
    results_json = {
        'image': image_path,
        'num_bubbles': len(detections),
        'bubbles': []
    }
    
    for i, (detection, ocr_result) in enumerate(zip(detections, ocr_results)):
        bubble_info = {
            'id': i + 1,
            'bbox': detection['bbox'],
            'yolo_confidence': detection['confidence'],
            'class_name': detection['class_name'],
            'text': ocr_result['text'],
            'text_lines': [line['text'] for line in ocr_result['lines']],
            'ocr_confidence': ocr_result['confidence']
        }
        results_json['bubbles'].append(bubble_info)
    
    json_path = result_dir / 'results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
    saved_files['json'] = str(json_path)
    
    return saved_files


# ============= 6. 主流程模块 =============

def process_image(image_path: str, yolo_model: YOLO, ocr_engine, config: Dict) -> Dict:
    """
    处理单张图片：检测气泡 + OCR识别
    
    Args:
        image_path: 图片路径
        yolo_model: YOLO模型
        ocr_engine: OCR引擎
        config: 配置参数
        
    Returns:
        处理结果
    """
    print(f"\n{'='*60}")
    print(f"处理图片: {image_path}")
    print(f"{'='*60}")
    
    total_start = time.time()
    timing = {}
    
    # 1. YOLO检测气泡
    step_start = time.time()
    detections = detect_bubbles(
        yolo_model, 
        image_path,
        conf=config['yolo_conf'],
        iou=config['yolo_iou']
    )
    timing['yolo_detection'] = time.time() - step_start
    
    if len(detections) == 0:
        print("未检测到气泡")
        return {
            'detections': [],
            'ocr_results': [],
            'saved_files': None
        }
    
    # 1.5 按位置排序（从上到下，考虑左右分布）
    step_start = time.time()
    # 对聊天气泡进行排序，确保从上到下的顺序
    # 使用 Y 坐标的中心点进行排序，这样更准确
    detections = sorted(detections, key=lambda x: (
        x['bbox'][1],  # 主要按 Y 坐标（上到下）
        x['bbox'][0]   # 次要按 X 坐标（左到右）
    ))
    timing['sorting'] = time.time() - step_start
    print(f"✅ 检测结果已按从上到下顺序排序 (共 {len(detections)} 个气泡)")
    
    # 读取图片
    image = cv2.imread(image_path)
    
    # 2. 批量裁剪所有气泡区域
    step_start = time.time()
    print(f"\n裁剪气泡区域...")
    cropped_images = []
    for detection in detections:
        bbox = detection['bbox']
        cropped = crop_bubble(image, bbox, config['padding'])
        cropped_images.append(cropped)
    timing['cropping'] = time.time() - step_start
    print(f"✅ 已裁剪 {len(cropped_images)} 个气泡区域")
    
    # 3. 批量 OCR 识别
    step_start = time.time()
    use_multithread = config.get('use_multithread', False)
    num_workers = config.get('num_workers', 4)
    
    if use_multithread and config['ocr_backend'] == 'tesseract':
        print(f"\n批量 OCR 识别（多线程，{num_workers} 个线程）...")
    else:
        print(f"\n批量 OCR 识别（单线程）...")
    
    ocr_results = recognize_text_batch(
        ocr_engine,
        cropped_images,
        backend=config['ocr_backend'],
        min_confidence=config['min_text_confidence'],
        preprocess=True,
        use_multithread=use_multithread,
        num_workers=num_workers
    )
    timing['ocr_recognition'] = time.time() - step_start
    
    # 4. 打印识别结果
    print(f"\n识别结果:")
    for i, ocr_result in enumerate(ocr_results):
        if ocr_result['text']:
            text_preview = ocr_result['text'].replace('\n', ' ')[:50]
            print(f"  气泡 {i+1}/{len(ocr_results)}: '{text_preview}' (置信度: {ocr_result['confidence']:.2f})")
        else:
            print(f"  气泡 {i+1}/{len(ocr_results)}: 未识别到文字")
    
    # 5. 保存结果
    step_start = time.time()
    print(f"\n保存结果...")
    saved_files = save_results(image_path, detections, ocr_results, config)
    timing['saving'] = time.time() - step_start
    
    timing['total'] = time.time() - total_start
    
    print(f"\n✅ 处理完成!")
    print(f"   检测到气泡: {len(detections)} 个")
    print(f"   识别到文字: {sum(1 for r in ocr_results if r['text'])} 个")
    print(f"   结果保存至: {saved_files['result_dir']}")
    
    # 打印耗时统计
    print(f"\n⏱️  耗时统计:")
    print(f"   YOLO检测:    {timing['yolo_detection']*1000:>7.2f} ms")
    print(f"   排序:        {timing['sorting']*1000:>7.2f} ms")
    print(f"   裁剪:        {timing['cropping']*1000:>7.2f} ms")
    print(f"   OCR识别:     {timing['ocr_recognition']*1000:>7.2f} ms  ⭐")
    print(f"   保存结果:    {timing['saving']*1000:>7.2f} ms")
    print(f"   {'─'*40}")
    print(f"   总耗时:      {timing['total']*1000:>7.2f} ms")
    
    if len(detections) > 0:
        print(f"\n   平均每个气泡: {timing['ocr_recognition']*1000/len(detections):.2f} ms")
    
    return {
        'detections': detections,
        'ocr_results': ocr_results,
        'saved_files': saved_files
    }


def main():
    """
    主函数
    """
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "气泡检测 + OCR 识别系统" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # 1. 加载 YOLO 模型
    try:
        yolo_model = load_yolo_model(CONFIG['yolo_model'])
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("\n请先训练模型:")
        print("  python train_yolo11n.py")
        return
    
    # 2. 初始化 OCR 引擎
    try:
        ocr_engine = init_ocr_engine(
            backend=CONFIG['ocr_backend'],
            lang=CONFIG['ocr_lang']
        )
    except Exception as e:
        print(f"\n❌ OCR引擎初始化失败: {e}")
        return
    
    # 3. 处理图片
    # 可以是单张图片或目录
    test_images = [
        # '/workspace/yolo/data/val/images/14d5a4c9-3f03e5d7-bb2d-4dc7-ab76-696bcc96bacf.jpg',
        '/workspace/yolo/bumble.jpg',
        # 添加更多测试图片
    ]
    
    results = []
    for image_path in test_images:
        if Path(image_path).exists():
            result = process_image(image_path, yolo_model, ocr_engine, CONFIG)
            results.append(result)
        else:
            print(f"\n⚠️  图片不存在: {image_path}")
    
    print(f"\n{'='*60}")
    print(f"所有图片处理完成！共处理 {len(results)} 张图片")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()


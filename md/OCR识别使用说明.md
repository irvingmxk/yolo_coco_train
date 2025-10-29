# 气泡检测 + OCR 文字识别使用说明

## 功能概述

本系统实现了**两步识别流程**：
1. **YOLO 检测**：使用 YOLO11n 模型检测图片中的气泡区域
2. **OCR 识别**：对检测到的每个气泡进行文字识别（支持多行文本）

## 模块架构

系统采用模块化设计，每个功能都是独立的函数：

### 核心模块

1. **YOLO 检测模块** (`bubble_ocr.py`)
   - `load_yolo_model()` - 加载 YOLO 模型
   - `detect_bubbles()` - 检测气泡区域

2. **图像预处理模块**
   - `crop_bubble()` - 裁剪气泡区域
   - `preprocess_for_ocr()` - OCR 预处理（去噪、二值化等）

3. **OCR 识别模块**
   - `init_ocr_engine()` - 初始化 OCR 引擎
   - `recognize_text()` - 通用识别接口
   - `recognize_text_tesseract()` - Tesseract 识别
   - `recognize_text_paddleocr()` - PaddleOCR 识别
   - `recognize_text_easyocr()` - EasyOCR 识别

4. **结果可视化模块**
   - `draw_results()` - 绘制检测框和识别文字

5. **结果保存模块**
   - `save_results()` - 保存裁剪图片、标注图片和 JSON 结果

6. **主流程模块**
   - `process_image()` - 完整处理单张图片的流程

## 快速开始

### 1. 安装 Tesseract-OCR（推荐）

Tesseract 是最简单的 OCR 方案，依赖少，安装简单：

```bash
# 运行安装脚本
bash install_tesseract.sh

# 或手动安装
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng

# CentOS/RHEL
sudo yum install tesseract tesseract-langpack-chi_sim tesseract-langpack-eng

# Python 包
pip install pytesseract pillow
```

### 2. 处理图片

```bash
# 激活环境
conda activate yolo

# 处理单张图片
python run_ocr.py --image your_image.jpg

# 批量处理目录
python run_ocr.py --dir data/val/images

# 指定 YOLO 模型
python run_ocr.py --image test.jpg --model runs/train/yolo11n_bubble/weights/best.pt

# 只识别英文
python run_ocr.py --image test.jpg --lang eng

# 调整检测置信度
python run_ocr.py --image test.jpg --conf 0.5
```

## OCR 引擎对比

| 引擎 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **Tesseract** | 安装简单，依赖少，速度快 | 识别率中等 | 快速开发，资源受限 |
| **PaddleOCR** | 识别率高，中文效果好 | 依赖多，安装复杂 | 生产环境，高精度要求 |
| **EasyOCR** | 多语言支持好 | 速度较慢，模型大 | 多语言场景 |

## 使用不同的 OCR 引擎

### Tesseract（默认，推荐）

```bash
# 默认使用 Tesseract
python run_ocr.py --image test.jpg

# 指定语言
python run_ocr.py --image test.jpg --lang chi_sim+eng  # 中英文
python run_ocr.py --image test.jpg --lang eng          # 仅英文
python run_ocr.py --image test.jpg --lang chi_sim      # 仅中文
```

### PaddleOCR

```bash
# 安装 PaddleOCR
pip install paddlepaddle-gpu paddleocr  # GPU版本
# 或
pip install paddlepaddle paddleocr      # CPU版本

# 使用 PaddleOCR
python run_ocr.py --image test.jpg --ocr paddleocr --lang ch
```

### EasyOCR

```bash
# 安装 EasyOCR
pip install easyocr

# 使用 EasyOCR
python run_ocr.py --image test.jpg --ocr easyocr --lang ch
```

## 输出结果

处理完成后，结果保存在 `runs/ocr/图片名/` 目录：

```
runs/ocr/your_image/
├── crops/                    # 裁剪的气泡图片
│   ├── bubble_1.jpg
│   ├── bubble_2.jpg
│   └── ...
├── annotated.jpg             # 标注后的图片（带检测框和文字）
└── results.json              # JSON 格式的详细结果
```

### JSON 结果格式

```json
{
  "image": "原图路径",
  "num_bubbles": 3,
  "bubbles": [
    {
      "id": 1,
      "bbox": [x1, y1, x2, y2],
      "yolo_confidence": 0.95,
      "class_name": "bubble",
      "text": "识别到的文字\n多行文本",
      "text_lines": ["识别到的文字", "多行文本"],
      "ocr_confidence": 0.87
    }
  ]
}
```

## Python API 使用

如果需要在自己的代码中使用，可以直接导入模块：

```python
from bubble_ocr import (
    load_yolo_model,
    init_ocr_engine,
    detect_bubbles,
    crop_bubble,
    recognize_text,
    process_image
)
import cv2

# 1. 加载模型
yolo_model = load_yolo_model('runs/train/yolo11n_bubble/weights/best.pt')
ocr_engine = init_ocr_engine(backend='tesseract', lang='chi_sim+eng')

# 2. 检测气泡
detections = detect_bubbles(yolo_model, 'test.jpg', conf=0.25)

# 3. 读取图片
image = cv2.imread('test.jpg')

# 4. 对每个气泡进行 OCR
for detection in detections:
    # 裁剪
    cropped = crop_bubble(image, detection['bbox'], padding=5)
    
    # 识别
    result = recognize_text(
        ocr_engine, 
        cropped, 
        backend='tesseract',
        min_confidence=0.5
    )
    
    print(f"识别文字: {result['text']}")
    print(f"置信度: {result['confidence']:.2f}")
```

## 完整流程示例

```python
from bubble_ocr import process_image, load_yolo_model, init_ocr_engine, CONFIG

# 配置
CONFIG['yolo_model'] = 'runs/train/yolo11n_bubble/weights/best.pt'
CONFIG['ocr_backend'] = 'tesseract'
CONFIG['ocr_lang'] = 'chi_sim+eng'
CONFIG['yolo_conf'] = 0.25

# 加载模型
yolo_model = load_yolo_model(CONFIG['yolo_model'])
ocr_engine = init_ocr_engine(
    backend=CONFIG['ocr_backend'],
    lang=CONFIG['ocr_lang']
)

# 处理图片
result = process_image('test.jpg', yolo_model, ocr_engine, CONFIG)

# 查看结果
print(f"检测到 {len(result['detections'])} 个气泡")
for i, (det, ocr) in enumerate(zip(result['detections'], result['ocr_results'])):
    print(f"\n气泡 {i+1}:")
    print(f"  位置: {det['bbox']}")
    print(f"  文字: {ocr['text']}")
```

## 参数配置

在 `bubble_ocr.py` 中的 `CONFIG` 字典可以调整各种参数：

```python
CONFIG = {
    # YOLO配置
    'yolo_model': 'runs/train/yolo11n_bubble/weights/best.pt',
    'yolo_conf': 0.25,        # 检测置信度阈值
    'yolo_iou': 0.45,         # NMS IOU阈值
    
    # OCR配置
    'ocr_backend': 'tesseract',
    'ocr_lang': 'chi_sim+eng',
    
    # 图像处理
    'padding': 5,             # 裁剪边距（像素）
    'min_text_confidence': 0.5,  # OCR最小置信度
    
    # 输出配置
    'save_crops': True,       # 保存裁剪图片
    'save_annotated': True,   # 保存标注图片
    'output_dir': 'runs/ocr', # 输出目录
}
```

## 多行文本处理

系统已经支持多行文本识别：

- OCR 会自动识别每一行文本
- 多行文本用换行符 `\n` 分隔
- 在结果中同时提供完整文本和按行分割的文本列表

```python
# 识别结果示例
{
    'text': '第一行文字\n第二行文字\n第三行文字',
    'lines': [
        {'text': '第一行文字', 'confidence': 0.9},
        {'text': '第二行文字', 'confidence': 0.85},
        {'text': '第三行文字', 'confidence': 0.88}
    ],
    'confidence': 0.88  # 平均置信度
}
```

## 优化建议

### 提高识别准确率

1. **调整图像预处理**：
   - 修改 `preprocess_for_ocr()` 函数
   - 调整去噪、二值化参数

2. **增加裁剪边距**：
   ```python
   CONFIG['padding'] = 10  # 增加到10像素
   ```

3. **降低置信度阈值**：
   ```python
   CONFIG['min_text_confidence'] = 0.3  # 降低到0.3
   ```

4. **使用更高质量的图片**

### 提高处理速度

1. **使用 GPU**：
   - Tesseract 和 PaddleOCR 都支持 GPU 加速

2. **批量处理时复用模型**：
   - 只加载一次模型
   - 循环处理多张图片

3. **调整 YOLO 参数**：
   ```python
   CONFIG['yolo_conf'] = 0.5  # 提高阈值，减少检测框数量
   ```

## 常见问题

### 1. Tesseract 找不到

```bash
# 检查是否安装
tesseract --version

# 查看支持的语言
tesseract --list-langs
```

### 2. 识别不到中文

```bash
# 安装中文语言包
sudo apt-get install tesseract-ocr-chi-sim
```

### 3. 内存不足

```python
# 减小批处理大小
# 或降低图像分辨率
CONFIG['padding'] = 2  # 减小裁剪边距
```

### 4. 识别率低

- 尝试不同的 OCR 引擎
- 调整图像预处理参数
- 使用更高质量的训练数据重新训练 YOLO 模型

## 完整示例

参考 `bumble.jpg` 图片的处理示例：

```bash
# 使用训练好的模型处理
python run_ocr.py \
    --image data/val/images/your_image.jpg \
    --model runs/train/yolo11n_bubble/weights/best.pt \
    --ocr tesseract \
    --lang chi_sim+eng \
    --conf 0.25
```

## 扩展功能

如需添加新的 OCR 引擎或功能，可以：

1. 在 `bubble_ocr.py` 中添加新的识别函数
2. 在 `init_ocr_engine()` 中添加初始化代码
3. 在 `recognize_text()` 中添加调用逻辑

## 技术支持

如有问题，请检查：
1. YOLO 模型是否训练完成
2. Tesseract 是否正确安装
3. 图片格式是否支持
4. 依赖包是否完整安装


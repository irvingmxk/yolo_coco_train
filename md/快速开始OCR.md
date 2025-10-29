# 气泡检测 + OCR 识别 快速开始

## ✅ 系统已就绪

所有组件已安装完成，可以直接使用！

## 快速测试

### 1. 处理单张图片

```bash
# 激活环境
conda activate yolo

# 处理单张图片（使用验证集中的第一张）
python run_ocr.py --image data/val/images/14d5a4c9-3f03e5d7-bb2d-4dc7-ab76-696bcc96bacf.jpg
```

### 2. 批量处理

```bash
# 批量处理验证集的所有图片
python run_ocr.py --dir data/val/images
```

## 结果查看

处理完成后，结果保存在 `runs/ocr/图片名/` 目录：

```
runs/ocr/你的图片名/
├── crops/              # 裁剪的气泡图片
│   ├── bubble_1.jpg   # 第1个气泡
│   ├── bubble_2.jpg   # 第2个气泡
│   └── ...
├── annotated.jpg       # 标注后的图片（带检测框和识别文字）
└── results.json        # JSON格式的详细结果
```

## 工作流程

1. **YOLO 检测**：检测图片中的气泡区域
2. **区域裁剪**：裁剪每个气泡区域（带边距）
3. **OCR 识别**：识别每个气泡中的文字（支持多行）
4. **结果保存**：保存裁剪图片、标注图片和 JSON 结果

## 模块说明

### 核心模块（bubble_ocr.py）

| 模块 | 功能 | 说明 |
|------|------|------|
| `load_yolo_model()` | 加载 YOLO 模型 | 加载训练好的检测模型 |
| `detect_bubbles()` | 检测气泡 | 返回所有气泡的位置和置信度 |
| `crop_bubble()` | 裁剪区域 | 根据检测框裁剪图片 |
| `preprocess_for_ocr()` | 图像预处理 | 去噪、二值化等 |
| `init_ocr_engine()` | 初始化 OCR | 加载 OCR 引擎 |
| `recognize_text()` | 文字识别 | 识别图片中的文字 |
| `draw_results()` | 绘制结果 | 在图片上标注检测框和文字 |
| `save_results()` | 保存结果 | 保存所有输出文件 |
| `process_image()` | 完整流程 | 执行完整的检测+识别流程 |

### 命令行工具（run_ocr.py）

简化的命令行接口，支持：
- 单张图片处理
- 批量目录处理
- 灵活的参数配置

## 高级用法

### 调整参数

```bash
# 提高检测置信度阈值（减少误检）
python run_ocr.py --image test.jpg --conf 0.5

# 只识别英文
python run_ocr.py --image test.jpg --lang eng

# 指定不同的模型
python run_ocr.py --image test.jpg --model path/to/your/model.pt
```

### Python API

```python
from bubble_ocr import process_image, load_yolo_model, init_ocr_engine, CONFIG

# 配置
CONFIG['yolo_conf'] = 0.25
CONFIG['min_text_confidence'] = 0.5

# 加载模型
yolo_model = load_yolo_model('runs/train/yolo11n_bubble/weights/best.pt')
ocr_engine = init_ocr_engine(backend='tesseract', lang='chi_sim+eng')

# 处理图片
result = process_image('test.jpg', yolo_model, ocr_engine, CONFIG)

# 查看结果
for i, (det, ocr) in enumerate(zip(result['detections'], result['ocr_results'])):
    print(f"气泡 {i+1}: {ocr['text']}")
```

## 多行文本处理

系统自动处理多行文本：

- 每行文本自动识别
- 行与行之间用 `\n` 分隔
- 结果中包含完整文本和分行文本

**示例输出：**
```json
{
  "text": "第一行文字\n第二行文字\n第三行文字",
  "lines": [
    {"text": "第一行文字", "confidence": 0.9},
    {"text": "第二行文字", "confidence": 0.85},
    {"text": "第三行文字", "confidence": 0.88}
  ],
  "confidence": 0.88
}
```

## 参数说明

### CONFIG 配置（bubble_ocr.py）

```python
CONFIG = {
    # YOLO 配置
    'yolo_model': 'runs/train/yolo11n_bubble/weights/best.pt',
    'yolo_conf': 0.25,      # 检测置信度阈值（0-1）
    'yolo_iou': 0.45,       # NMS IOU 阈值
    
    # OCR 配置
    'ocr_backend': 'tesseract',     # OCR 引擎
    'ocr_lang': 'chi_sim+eng',      # 语言
    
    # 图像处理
    'padding': 5,                    # 裁剪边距（像素）
    'min_text_confidence': 0.5,     # OCR 最小置信度
    
    # 输出
    'save_crops': True,              # 保存裁剪图片
    'save_annotated': True,          # 保存标注图片
    'output_dir': 'runs/ocr',       # 输出目录
}
```

## 性能优化建议

### 提高识别准确率

1. **增加裁剪边距**
   ```python
   CONFIG['padding'] = 10  # 增加到10像素
   ```

2. **降低置信度阈值**（获取更多文本）
   ```python
   CONFIG['min_text_confidence'] = 0.3
   ```

3. **使用高质量图片**

### 提高处理速度

1. **提高检测阈值**（减少检测框）
   ```python
   CONFIG['yolo_conf'] = 0.5
   ```

2. **批量处理时复用模型**（避免重复加载）

## 完整示例

```bash
# 1. 激活环境
conda activate yolo

# 2. 处理图片
python run_ocr.py --image data/val/images/your_image.jpg

# 3. 查看结果
cat runs/ocr/your_image/results.json

# 4. 查看标注图片
# runs/ocr/your_image/annotated.jpg

# 5. 查看裁剪的气泡
# runs/ocr/your_image/crops/bubble_*.jpg
```

## 技术特点

✅ **模块化设计**：每个功能都是独立函数  
✅ **支持多行文本**：自动处理多行文本识别  
✅ **简单易用**：Tesseract 安装简单，依赖少  
✅ **灵活配置**：所有参数都可调整  
✅ **完整输出**：保存裁剪图、标注图、JSON 结果  
✅ **可扩展**：易于添加新的 OCR 引擎  

## 相关文档

- **详细文档**：`md/OCR识别使用说明.md`
- **YOLO训练**：`md/YOLO11n训练说明.md`
- **快速参考**：`md/快速参考.md`

## 常见问题

### Q: 识别不到文字？
A: 尝试降低 `min_text_confidence` 阈值，或增加 `padding` 边距

### Q: 识别率低？
A: 尝试使用 PaddleOCR（需要额外安装），或提高图片质量

### Q: 内存不足？
A: 减小 `padding` 值，或逐张处理图片

## 系统状态

- ✅ YOLO 模型已训练
- ✅ Tesseract-OCR 已安装
- ✅ 中英文语言包已安装
- ✅ 测试数据已准备
- ✅ 所有模块正常

**🎉 可以开始使用了！**


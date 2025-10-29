# YOLO11n 气泡检测项目

基于 YOLO11n 模型的气泡检测系统，使用迁移学习进行微调训练。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 使用详细配置的训练脚本（推荐）
python train_yolo11n.py

# 或使用简化版训练脚本
python train.py
```

### 3. 模型推理

```bash
python inference.py
```

## 项目文件说明

- `train_yolo11n.py` - YOLO11n 详细训练脚本（推荐）
- `train.py` - 简化版训练脚本
- `inference.py` - 模型推理脚本
- `yolo11n.pt` - YOLO11n 预训练模型
- `data/` - 训练数据集
- `runs/` - 训练输出结果
- `md/` - 详细文档

## 详细文档

请查看 `md/YOLO11n训练说明.md` 获取完整的使用说明。

## 数据集

- 类别：bubble（气泡）
- 训练集：data/train/
- 验证集：data/val/

## 模型性能

训练完成后，模型权重保存在：
- `runs/train/yolo11n_bubble/weights/best.pt`

## 技术栈

- Python 3.8+
- PyTorch 2.0+
- Ultralytics YOLO11n
- OpenCV

## License

MIT

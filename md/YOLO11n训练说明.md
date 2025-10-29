# YOLO11n 气泡检测训练说明

## 项目概述

本项目使用 YOLO11n 模型进行气泡检测的迁移学习训练。

## 项目结构

```
yolo/
├── train_yolo11n.py          # YOLO11n训练脚本（推荐使用）
├── train.py                   # 简化版训练脚本
├── inference.py               # 推理脚本
├── yolo11n.pt                 # YOLO11n预训练模型
├── requirements.txt           # Python依赖包
├── data/                      # 数据集目录
│   ├── data.yaml             # 数据集配置文件
│   ├── classes.txt           # 类别列表
│   ├── train/                # 训练集
│   │   ├── images/          # 训练图片
│   │   └── labels/          # 训练标签
│   └── val/                  # 验证集
│       ├── images/          # 验证图片
│       └── labels/          # 验证标签
└── runs/                      # 训练输出目录
    └── train/                # 训练结果
        └── yolo11n_bubble/   # YOLO11n训练结果
            └── weights/      # 模型权重
                ├── best.pt   # 最佳模型
                └── last.pt   # 最新模型
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (可选，用于GPU加速)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 训练模型

### 方法1: 使用详细配置的训练脚本（推荐）

```bash
python train_yolo11n.py
```

**特点：**
- 完整的训练配置参数
- 详细的环境检查
- 自动验证模型性能
- 更好的训练策略（余弦学习率、warmup等）

### 方法2: 使用简化版训练脚本

```bash
python train.py
```

**特点：**
- 代码简洁
- 适合快速训练
- 自动分割数据集（如果还未分割）

## 训练配置说明

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | `yolo11n.pt` | 预训练模型 |
| `epochs` | `100` | 训练轮数 |
| `batch` | `16` | 批次大小 |
| `imgsz` | `640` | 输入图像大小 |
| `device` | `0` | GPU设备（0,1,2... 或 'cpu'） |
| `patience` | `50` | 早停等待轮数 |
| `lr0` | `0.01` | 初始学习率 |

### 修改配置

编辑 `train_yolo11n.py` 文件中的 `CONFIG` 字典：

```python
CONFIG = {
    'model': 'yolo11n.pt',
    'epochs': 100,        # 修改训练轮数
    'batch': 16,          # 根据显存调整批次大小
    'device': 0,          # GPU设备编号
    # ... 其他参数
}
```

### 批次大小建议

根据显存选择合适的批次大小：

- 4GB 显存：batch=8
- 8GB 显存：batch=16
- 16GB+ 显存：batch=32

## 模型推理

训练完成后，使用推理脚本进行预测：

```bash
python inference.py
```

推理脚本会自动：
1. 查找训练好的最佳模型
2. 对验证集图片进行预测
3. 保存预测结果到 `runs/predict/results/`

## 训练结果

训练完成后，结果保存在：

```
runs/train/yolo11n_bubble/
├── weights/
│   ├── best.pt                          # 最佳模型（mAP最高）
│   └── last.pt                          # 最新模型
├── results.png                          # 训练曲线
├── results.csv                          # 训练数据
├── confusion_matrix.png                 # 混淆矩阵
├── confusion_matrix_normalized.png      # 归一化混淆矩阵
├── BoxP_curve.png                       # Precision曲线
├── BoxR_curve.png                       # Recall曲线
├── BoxF1_curve.png                      # F1曲线
└── BoxPR_curve.png                      # PR曲线
```

## 评估指标说明

- **mAP50**: 在IoU=0.5时的平均精度（主要指标）
- **mAP50-95**: IoU从0.5到0.95的平均精度
- **Precision**: 精确率（预测为正的样本中真正为正的比例）
- **Recall**: 召回率（真正为正的样本中被预测为正的比例）

## 常见问题

### 1. CUDA out of memory（显存不足）

**解决方法：**
- 减小 `batch` 大小
- 减小 `imgsz` 图像大小
- 使用更小的模型（yolo11n已经是最小的）

### 2. 训练速度慢

**解决方法：**
- 确保使用GPU训练（device=0）
- 增加 `workers` 数量
- 减小验证频率

### 3. 模型不收敛

**解决方法：**
- 检查数据标注是否正确
- 调整学习率
- 增加训练轮数
- 调整数据增强参数

### 4. 过拟合

**解决方法：**
- 增加数据增强
- 增加训练数据
- 调整 `weight_decay`
- 使用早停策略（已默认启用）

## 高级用法

### 恢复训练

如果训练中断，可以从最后一个检查点恢复：

```python
model = YOLO('runs/train/yolo11n_bubble/weights/last.pt')
model.train(resume=True)
```

### 使用不同的优化器

在 `CONFIG` 中修改：

```python
'optimizer': 'AdamW',  # 可选：SGD, Adam, AdamW, RMSProp
```

### 多GPU训练

```python
'device': [0, 1, 2, 3],  # 使用4个GPU
```

## 使用预训练模型进行推理

如果只想使用已训练好的模型：

```bash
# 指定模型路径
python inference.py
```

## 技术支持

如有问题，请检查：
1. 数据集格式是否正确
2. 依赖包是否安装完整
3. CUDA环境是否配置正确

## 更新日志

- **2025-10-28**: 创建项目，使用YOLO11n模型
- 支持气泡检测任务
- 完整的训练和推理流程


# 聊天记录保存和智能合并功能说明

## 功能概述

在原有的 `bubble_ocr.py` 基础上，新增了聊天记录保存和智能合并功能：

1. **第一次上传图片**：使用 OCR 识别后，直接保存聊天记录
2. **第二次及以后上传**：使用 OCR 识别后，智能合并新的聊天记录到已有记录中

## 文件结构

```
yolo/
├── bubble_ocr.py              # 主程序（已集成聊天记录功能）
├── chat_record_manager.py     # 聊天记录管理模块（独立文件）
├── chat_record_usage_example.py  # 使用示例
└── CHAT_RECORD_README.md      # 本说明文档
```

## 核心功能

### 1. 聊天记录管理模块 (`chat_record_manager.py`)

提供以下功能：
- **智能合并**：使用 LCS（最长公共子序列）算法合并聊天记录
- **去重检测**：快速检测完全相同或子集的情况
- **OCR容错**：针对 OCR 识别错误，降低短文本相似度阈值
- **原子写入**：使用临时文件确保文件写入的原子性
- **内存缓存**：使用内存缓存提高性能，同时支持从文件恢复

### 2. 集成到 `bubble_ocr.py`

- 自动在 OCR 识别后保存聊天记录
- 支持通过 `session_id` 区分不同的聊天会话
- 返回合并信息（新增消息数、累计总数、操作类型）

## 使用方法

### 基本使用

```python
from bubble_ocr import process_image, load_yolo_model, init_ocr_engine, CONFIG

# 加载模型
yolo_model = load_yolo_model(CONFIG['yolo_model'])
ocr_engine = init_ocr_engine(
    backend=CONFIG['ocr_backend'],
    lang=CONFIG['ocr_lang']
)

# 处理第一张图片（使用 session_id）
session_id = "chat_session_001"
result1 = process_image(
    "/path/to/image1.jpg",
    yolo_model,
    ocr_engine,
    CONFIG,
    session_id=session_id
)

# 处理第二张图片（使用相同的 session_id 进行合并）
result2 = process_image(
    "/path/to/image2.jpg",
    yolo_model,
    ocr_engine,
    CONFIG,
    session_id=session_id  # 相同的 session_id
)

# 查看合并信息
if result2.get('merge_info'):
    merge_info = result2['merge_info']
    print(f"新增消息: {merge_info['added_count']} 条")
    print(f"累计总数: {merge_info['total_count']} 条")
    print(f"操作类型: {merge_info['operation_type']}")
```

### 获取保存的聊天记录

```python
from chat_record_manager import chat_record_manager

session_id = "chat_session_001"

# 获取文本格式
chat_text = chat_record_manager.get_chat_text(session_id)
print(chat_text)

# 获取结构化数据
records = chat_record_manager.get_chat_records(session_id)
for record in records:
    print(f"[{record['speaker']}] {record['content']}")
```

### 配置选项

在 `bubble_ocr.py` 的 `CONFIG` 中：

```python
CONFIG = {
    # ... 其他配置 ...
    
    # 聊天记录保存配置
    'save_chat_records': True,      # 是否保存聊天记录
    'session_id': None,             # 会话ID（None则从图片名称自动生成）
}
```

## 保存位置

聊天记录保存在：
```
./chat_records/{session_id}/
├── chat_records.txt    # 文本格式（每行: speaker: content）
└── chat_records.json   # JSON格式（包含元数据）
```

## 智能合并逻辑

### 1. 快速检测
- **完全相同**：如果新记录与已有记录完全相同，直接跳过
- **子集检测**：如果新记录是已有记录的子集，跳过

### 2. 智能合并（LCS算法）
- 使用最长公共子序列算法找到重叠部分
- 自动去重，只保留新增的消息
- 选择更完整的消息版本（处理 OCR 识别差异）

### 3. OCR容错
- 短文本（<10字符）：相似度阈值 0.80
- 中等文本（10-30字符）：相似度阈值 0.82
- 长文本（>30字符）：相似度阈值 0.85

## 操作类型说明

- `initial`：第一次保存
- `no_new`：没有新记录
- `identical_skip`：完全相同，跳过
- `subset_skip`：子集，跳过
- `global_align_merge`：智能合并成功
- `error`：合并失败

## 注意事项

1. **session_id**：相同 `session_id` 的记录会被合并，不同 `session_id` 的记录独立保存
2. **内存缓存**：程序运行时会使用内存缓存，服务重启后会自动从文件恢复
3. **文件格式**：保存为文本格式便于查看，同时保存 JSON 格式便于后续处理

## 示例场景

### 场景1：连续截图同一聊天
```python
# 第一页截图
process_image("page1.jpg", ..., session_id="chat_001")

# 第二页截图（包含部分第一页的内容）
process_image("page2.jpg", ..., session_id="chat_001")
# 会自动合并，只保存新增的消息
```

### 场景2：多个独立聊天
```python
# 聊天A
process_image("chatA_page1.jpg", ..., session_id="chat_A")
process_image("chatA_page2.jpg", ..., session_id="chat_A")

# 聊天B（独立）
process_image("chatB_page1.jpg", ..., session_id="chat_B")
```

## 性能说明

- **内存缓存**：首次加载后，后续操作直接使用内存，速度很快
- **原子写入**：使用临时文件 + `os.replace()` 确保文件完整性
- **合并算法**：时间复杂度 O(n*m)，对于常见场景（几十条消息）非常快


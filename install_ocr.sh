#!/bin/bash
# OCR 依赖安装脚本

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              安装 OCR 识别所需依赖包                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 检查conda环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  请先激活 conda 环境:"
    echo "   conda activate yolo"
    exit 1
fi

echo "✅ 当前环境: $CONDA_DEFAULT_ENV"
echo ""

# 询问用户选择OCR引擎
echo "请选择要安装的 OCR 引擎:"
echo "  1) PaddleOCR (推荐，支持中文，速度快)"
echo "  2) EasyOCR (支持多语言)"
echo "  3) 同时安装两个"
echo ""
read -p "请输入选择 (1/2/3): " choice

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

case $choice in
    1)
        echo "安装 PaddleOCR..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # 检查CUDA
        if command -v nvidia-smi &> /dev/null; then
            echo "✅ 检测到 NVIDIA GPU，安装 GPU 版本"
            pip install paddlepaddle-gpu paddleocr -i https://mirror.baidu.com/pypi/simple
        else
            echo "⚠️  未检测到 GPU，安装 CPU 版本"
            pip install paddlepaddle paddleocr -i https://mirror.baidu.com/pypi/simple
        fi
        ;;
        
    2)
        echo "安装 EasyOCR..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        pip install easyocr
        ;;
        
    3)
        echo "安装 PaddleOCR 和 EasyOCR..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # PaddleOCR
        if command -v nvidia-smi &> /dev/null; then
            echo "✅ 检测到 NVIDIA GPU，安装 PaddleOCR GPU 版本"
            pip install paddlepaddle-gpu paddleocr -i https://mirror.baidu.com/pypi/simple
        else
            echo "⚠️  未检测到 GPU，安装 PaddleOCR CPU 版本"
            pip install paddlepaddle paddleocr -i https://mirror.baidu.com/pypi/simple
        fi
        
        # EasyOCR
        echo ""
        echo "安装 EasyOCR..."
        pip install easyocr
        ;;
        
    *)
        echo "❌ 无效的选择"
        exit 1
        ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 安装完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "测试安装:"
echo "  python -c 'from paddleocr import PaddleOCR; print(\"✅ PaddleOCR OK\")'"
echo "  python -c 'import easyocr; print(\"✅ EasyOCR OK\")'"
echo ""
echo "开始使用:"
echo "  python run_ocr.py --image your_image.jpg"
echo ""


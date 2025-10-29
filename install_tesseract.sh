#!/bin/bash
# Tesseract-OCR 简易安装脚本

echo "╔════════════════════════════════════════════════════════════╗"
echo "║            安装 Tesseract-OCR (简单 OCR 方案)              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 检测操作系统
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "❌ 无法检测操作系统"
    exit 1
fi

echo "操作系统: $OS"
echo ""

# 安装 Tesseract-OCR
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. 安装 Tesseract-OCR 系统包"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

case $OS in
    ubuntu|debian)
        echo "使用 apt 安装..."
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng
        ;;
        
    centos|rhel|fedora)
        echo "使用 yum/dnf 安装..."
        sudo yum install -y tesseract tesseract-langpack-chi_sim tesseract-langpack-eng || \
        sudo dnf install -y tesseract tesseract-langpack-chi_sim tesseract-langpack-eng
        ;;
        
    *)
        echo "⚠️  未识别的操作系统: $OS"
        echo ""
        echo "请手动安装 Tesseract-OCR:"
        echo "  Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim"
        echo "  CentOS/RHEL:   sudo yum install tesseract tesseract-langpack-chi_sim"
        echo "  macOS:         brew install tesseract tesseract-lang"
        echo ""
        read -p "是否继续安装 Python 包? (y/n): " continue_install
        if [[ ! $continue_install =~ ^[Yy]$ ]]; then
            exit 1
        fi
        ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. 验证 Tesseract 安装"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v tesseract &> /dev/null; then
    echo "✅ Tesseract 已安装"
    tesseract --version | head -n 1
    echo ""
    echo "支持的语言:"
    tesseract --list-langs
else
    echo "❌ Tesseract 未安装成功"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. 安装 Python 包 (pytesseract)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 检查conda环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  建议在 conda 环境中安装:"
    echo "   conda activate yolo"
    echo ""
    read -p "是否继续在当前环境安装? (y/n): " install_now
    if [[ ! $install_now =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 0
    fi
else
    echo "✅ 当前环境: $CONDA_DEFAULT_ENV"
fi

echo ""
pip install pytesseract pillow

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. 测试安装"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python << 'EOF'
try:
    import pytesseract
    version = pytesseract.get_tesseract_version()
    print(f"✅ pytesseract 安装成功")
    print(f"   Tesseract 版本: {version}")
except Exception as e:
    print(f"❌ 测试失败: {e}")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Tesseract-OCR 安装完成！"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "开始使用:"
    echo "  # 处理单张图片"
    echo "  python run_ocr.py --image your_image.jpg"
    echo ""
    echo "  # 批量处理"
    echo "  python run_ocr.py --dir data/val/images"
    echo ""
else
    echo ""
    echo "❌ 安装失败，请检查错误信息"
    exit 1
fi


#!/bin/bash

# 性能分析报告工具安装脚本
echo "开始安装性能分析报告工具依赖..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 安装Python依赖
echo "安装Python依赖包..."
pip3 install -r requirements.txt

# 检查字体文件
if [ ! -f "./simsun.ttc" ]; then
    echo "警告: 未找到中文字体文件 simsun.ttc"
    echo "请手动下载宋体字体文件(simsun.ttc)并放置在当前目录下"
    echo "字体文件对于生成带有中文的图表和报告是必需的"
else
    echo "已找到中文字体文件 simsun.ttc"
fi

# 检查wkhtmltopdf（用于PDF生成）
if ! command -v wkhtmltopdf &> /dev/null; then
    echo "警告: 未找到wkhtmltopdf，这可能会影响PDF报告生成"
    echo "请安装wkhtmltopdf: apt-get install -y wkhtmltopdf"
fi

echo "安装完成！"
echo "使用方法:"
echo "1. 提取数据: python3 report.py --mode extract"
echo "2. 分析数据: python3 report.py --mode analyze"
echo "3. 生成报告: python3 report.py --mode report"
echo "详细说明请参考README.md文件"

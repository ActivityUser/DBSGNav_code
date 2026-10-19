#!/bin/bash
echo "🔄 创建 gausshabitat 环境"
echo "================================"

if [ -f "gausshabitat_nobuilds.yaml" ]; then
    echo "1. 从YAML文件创建环境..."
    conda env create -f gausshabitat_nobuilds.yaml
    
    echo "2. 激活环境..."
    conda activate gausshabitat
    
    echo "3. 安装pip包..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
    
    echo "✅ 创建恢复完成！"
    echo "   使用: conda activate gausshabitat"

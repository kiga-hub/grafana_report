#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
性能测试报告比较脚本

此脚本用于比较两个不同时间段的Grafana导出的CSV数据,并生成Markdown格式的性能对比报告。
脚本会自动识别grafana_csv目录下的日期文件夹,选择最新和次新的数据集进行比较分析。
生成的Markdown报告将自动转换为PDF文件，使用绝对路径确保图像正确加载，并确保中文显示正确。
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import re
import matplotlib.pyplot as plt
import markdown
import pdfkit
from analysis import load_data, analyze_data, setup_chinese_font

# 默认分析阈值配置
DEFAULT_THRESHOLDS = {
    "mean_median_ratio": 1.2,
    "max_mean_ratio": 2.0,
    "std_mean_ratio": 0.3,
    "network_std_mean_ratio": 0.5,
    "outlier_std_times": 3.0,
    "stability_good_percentage": 80.0,
    "stability_normal_percentage": 50.0,
    "resource_good_percentage": 0.7
}

# 指标类型关键词配置
METRIC_TYPE_KEYWORDS = {
    "cpu": ["cpu", "负载", "load"],
    "memory": ["内存", "memory", "ram"],
    "disk": ["磁盘", "disk", "iops", "io", "存储"],
    "network": ["网络", "network", "数据包", "packet", "流量", "traffic", "socket", "连接"]
}

def load_config(config_path=None):
    """加载自定义配置文件，如果不存在则使用默认配置"""
    config = {
        "thresholds": DEFAULT_THRESHOLDS.copy(),
        "metric_keywords": METRIC_TYPE_KEYWORDS.copy()
    }
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                
            if "thresholds" in user_config:
                config["thresholds"].update(user_config["thresholds"])
                
            if "metric_keywords" in user_config:
                for metric_type, keywords in user_config["metric_keywords"].items():
                    if metric_type in config["metric_keywords"]:
                        config["metric_keywords"][metric_type] = list(set(
                            config["metric_keywords"].get(metric_type, []) + keywords
                        ))
                    else:
                        config["metric_keywords"][metric_type] = keywords
                        
            print(f"已加载自定义配置: {config_path}")
        except Exception as e:
            print(f"加载配置文件出错: {e}，将使用默认配置")
    
    return config

def find_date_folders(base_dir):
    """在指定目录下查找日期格式的文件夹，并按日期排序"""
    if not os.path.exists(base_dir) or not os.path.isdir(base_dir):
        print(f"错误：目录不存在: {base_dir}")
        return []
    
    date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}-\d{2}:\d{2}$')
    date_folders = []
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and date_pattern.match(item):
            try:
                folder_date = datetime.strptime(item, '%Y-%m-%d-%H:%M')
                date_folders.append((item, folder_date))
            except ValueError:
                print(f"警告：文件夹名称 {item} 符合日期格式但无法解析")
    
    date_folders.sort(key=lambda x: x[1], reverse=True)
    return [folder[0] for folder in date_folders]

def get_comparison_folders(base_dir):
    """获取用于比较的两个文件夹（最新和次新）"""
    date_folders = find_date_folders(base_dir)
    
    if len(date_folders) < 2:
        print(f"错误：需要至少两个日期文件夹进行比较，但只找到 {len(date_folders)} 个")
        return None
    
    return (date_folders[0], date_folders[1])

def get_common_csv_files(folder1_path, folder2_path):
    """获取两个文件夹中共有的CSV文件列表"""
    if not os.path.exists(folder1_path) or not os.path.exists(folder2_path):
        return []
    
    files1 = set([f for f in os.listdir(folder1_path) if f.endswith('.csv')])
    files2 = set([f for f in os.listdir(folder2_path) if f.endswith('.csv')])
    return list(files1.intersection(files2))

def calculate_change_percentage(value1, value2):
    """计算两个值之间的变化百分比"""
    if value1 == 0:
        return float('inf') if value2 > 0 else 0.0
    return ((value2 - value1) / abs(value1)) * 100.0

def create_comparison_chart(metric_changes, output_path, title, font_prop=None):
    """创建新旧数据对比柱状图"""
    if not metric_changes:
        return
    
    metrics = list(metric_changes.keys())
    old_values = [metric_changes[m]["old_mean"] for m in metrics]
    new_values = [metric_changes[m]["new_mean"] for m in metrics]
    
    max_metrics = 10
    if len(metrics) > max_metrics:
        changes_abs = [abs(metric_changes[m]["change_pct"]) for m in metrics]
        indices = np.argsort(changes_abs)[-max_metrics:]
        metrics = [metrics[i] for i in indices]
        old_values = [old_values[i] for i in indices]
        new_values = [new_values[i] for i in indices]
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, old_values, width, label='较旧数据', alpha=0.7, color='#1f77b4')
    plt.bar(x + width/2, new_values, width, label='较新数据', alpha=0.7, color='#ff7f0e')
    
    for i, v in enumerate(old_values):
        plt.text(i - width/2, v + max(old_values + new_values) * 0.01, f'{v:.2f}', 
                 ha='center', va='bottom', fontsize=9, rotation=45, fontproperties=font_prop)
    for i, v in enumerate(new_values):
        plt.text(i + width/2, v + max(old_values + new_values) * 0.01, f'{v:.2f}', 
                 ha='center', va='bottom', fontsize=9, rotation=45, fontproperties=font_prop)
    
    for i, m in enumerate(metrics):
        change_pct = metric_changes[m]["change_pct"]
        change_symbol = "↑" if change_pct > 0 else "↓" if change_pct < 0 else ""
        color = 'red' if change_pct > 0 else 'green' if change_pct < 0 else 'black'
        plt.text(i, max(old_values[i], new_values[i]) + max(old_values + new_values) * 0.05, 
                 f'{abs(change_pct):.1f}% {change_symbol}', 
                 ha='center', va='bottom', fontsize=10, color=color, weight='bold', fontproperties=font_prop)
    
    plt.xlabel('指标', fontsize=12, fontproperties=font_prop)
    plt.ylabel('数值', fontsize=12, fontproperties=font_prop)
    plt.title(f'{title} - 新旧数据对比', fontsize=14, fontproperties=font_prop)
    plt.xticks(x, metrics, rotation=45, ha='right', fontproperties=font_prop)
    plt.legend(prop=font_prop)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

def create_summary_chart(type_changes, output_path, font_prop=None):
    """创建总体性能变化对比图表"""
    if not type_changes:
        return
    
    types = []
    avg_changes = []
    std_changes = []
    
    for metric_type, changes in type_changes.items():
        if not changes:
            continue
        
        change_values = [c[1] for c in changes]
        avg_change = sum(change_values) / len(change_values)
        std_change = np.std(change_values) if len(change_values) > 1 else 0
        
        display_type = {
            "CPU/负载": "CPU/负载",
            "内存": "内存",
            "磁盘/存储": "磁盘/存储",
            "网络": "网络"
        }.get(metric_type, metric_type.capitalize())
        
        types.append(display_type)
        avg_changes.append(avg_change)
        std_changes.append(std_change)
    
    if not types:
        return
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(types))
    
    colors = ['#ff7f0e' if c > 5 else '#2ca02c' if c < -5 else '#1f77b4' for c in avg_changes]
    plt.bar(x, avg_changes, width=0.6, alpha=0.7, color=colors)
    plt.errorbar(x, avg_changes, yerr=std_changes, fmt='none', ecolor='black', capsize=5, alpha=0.5)
    
    for i, v in enumerate(avg_changes):
        change_symbol = "↑" if v > 0 else "↓" if v < 0 else ""
        color = 'red' if v > 5 else 'green' if v < -5 else 'black'
        plt.text(i, v + (0.1 if v >= 0 else -2.5), f'{abs(v):.2f}% {change_symbol}', 
                 ha='center', va='bottom', fontsize=11, color=color, weight='bold', fontproperties=font_prop)
    
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='显著增加阈值 (5%)')
    plt.axhline(y=-5, color='green', linestyle='--', alpha=0.5, label='显著减少阈值 (-5%)')
    
    plt.xlabel('指标类型', fontsize=12, fontproperties=font_prop)
    plt.ylabel('平均变化百分比 (%)', fontsize=12, fontproperties=font_prop)
    plt.title('系统性能各指标类型平均变化', fontsize=14, fontproperties=font_prop)
    plt.xticks(x, types, fontproperties=font_prop)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(prop=font_prop)
    plt.ylim(min(plt.ylim()[0], -10), max(plt.ylim()[1], 15))
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

import markdown
import pdfkit
import os
import re
import urllib.parse

import markdown
import pdfkit
import os
import re
import urllib.parse

def convert_markdown_to_pdf(markdown_file, pdf_file, font_prop=None):
    """将Markdown文件转换为PDF文件，确保中文显示正确"""
    try:
        # Read Markdown content
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Convert relative image paths to absolute file:// paths
        base_path = os.path.dirname(os.path.abspath(markdown_file))
        def replace_image_path(match):
            alt_text = match.group(1)
            rel_path = match.group(2)
            # Construct absolute path
            abs_path = os.path.normpath(os.path.join(base_path, rel_path))
            # Verify image file exists
            if not os.path.exists(abs_path):
                print(f"警告: 图像文件不存在: {abs_path}")
                return f'![{alt_text}](about:blank)'
            # Convert to file:// URL with proper encoding for special characters
            file_url = 'file:///' + urllib.parse.quote(abs_path.replace(os.sep, '/'), safe='/')
            print(f"调试: 图像路径转换: {rel_path} -> {file_url}")
            return f'![{alt_text}]({file_url})'
        
        # Enhanced regex to handle paths with special characters and Chinese text
        markdown_content = re.sub(
            r'!\[(.*?)\]\(([^)]+)\)',
            replace_image_path,
            markdown_content,
            flags=re.UNICODE
        )
        
        # Convert Markdown to HTML with basic styling
        html_content = markdown.markdown(markdown_content, extensions=['extra', 'tables'])
        
        # Add CSS for better PDF rendering with font fallbacks
        css = """
        <style>
            @font-face {
                font-family: 'Noto Sans CJK SC';
                src: local('Noto Sans CJK SC'), local('NotoSansCJKsc-Regular');
                font-weight: normal;
            }
            body {
                font-family: 'Noto Sans CJK SC', 'Source Han Sans CN', 'PingFang SC', 'Microsoft YaHei', sans-serif;
                margin: 20px;
                line-height: 1.6;
            }
            h1 { font-size: 24px; color: #333; }
            h2 { font-size: 20px; color: #444; }
            h3 { font-size: 18px; color: #555; }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 10px 0;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th { background-color: #f2f2f2; }
            img { max-width: 100%; height: auto; display: block; }
            pre, code { background-color: #f4f4f4; padding: 5px; }
        </style>
        """
        html_content = f"<html><head><meta charset='UTF-8'>{css}</head><body>{html_content}</body></html>"
        
        # Save temporary HTML file
        temp_html = markdown_file.replace('.md', '.temp.html')
        with open(temp_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Specify wkhtmltopdf path explicitly
        wkhtmltopdf_path = '/usr/bin/wkhtmltopdf'
        configuration = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
        
        # Convert HTML to PDF with local file access enabled
        options = {
            'encoding': 'UTF-8',
            'enable-local-file-access': None,  # Allow file:// URLs
            'quiet': '',  # Suppress wkhtmltopdf console output
            'dpi': '300',  # Improve rendering quality
            'no-outline': None,  # Disable outline for cleaner PDF
            'disable-smart-shrinking': None,  # Prevent font rendering issues
            'load-error-handling': 'ignore',  # Ignore missing image errors
            'load-media-error-handling': 'ignore'  # Ignore media errors
        }
        pdfkit.from_file(temp_html, pdf_file, configuration=configuration, options=options)
        
        # Clean up temporary HTML file
        os.remove(temp_html)
        
        print(f"PDF报告生成成功: {pdf_file}")
        return True
    except Exception as e:
        print(f"转换Markdown到PDF时出错: {e}")
        print("请检查以下内容：")
        print("1. 确保 wkhtmltopdf 已正确安装：")
        print("   wkhtmltopdf --version")
        print("2. 确保图像文件存在且路径正确：")
        print(f"   ls -l {os.path.dirname(markdown_file)}/data/*/*.png")
        print("3. 确保有权限访问文件：")
        print(f"   chmod -R u+rw {os.path.dirname(markdown_file)}")
        print("4. 确保 pdfkit 版本兼容：")
        print("   pip install --upgrade pdfkit")
        print("5. 确保 Noto Sans CJK SC 字体已安装：")
        print("   fc-list :lang=zh | grep Noto")
        print("   sudo apt-get install -y fonts-noto-cjk")
        print("6. 尝试以下命令重新安装 wkhtmltopdf：")
        print("   sudo apt-get update")
        print("   sudo apt-get install -y wkhtmltopdf")
        print("   或者手动安装：")
        print("   wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6.1-2/wkhtmltox_0.12.6.1-2.jammy_amd64.deb")
        print("   sudo dpkg -i wkhtmltox_0.12.6.1-2.jammy_amd64.deb")
        print("   sudo apt-get install -f")
        return False
def convert_markdown_to_pdf4(markdown_file, pdf_file, font_prop=None):
    """将Markdown文件转换为PDF文件，确保中文显示正确"""
    try:
        # Read Markdown content
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Convert relative image paths to absolute file:// paths
        base_path = os.path.dirname(os.path.abspath(markdown_file))
        def replace_image_path(match):
            alt_text = match.group(1)
            rel_path = match.group(2)
            # Construct absolute path
            abs_path = os.path.abspath(os.path.join(base_path, rel_path))
            # Verify image file exists
            if not os.path.exists(abs_path):
                print(f"警告: 图像文件不存在: {abs_path}")
                return f'![{alt_text}](about:blank)'
            # Convert to file:// URL and ensure proper encoding
            file_url = 'file://' + urllib.parse.quote(os.path.normpath(abs_path))
            return f'![{alt_text}]({file_url})'
        
        markdown_content = re.sub(
            r'!\[(.*?)\]\((.*?)\)',
            replace_image_path,
            markdown_content
        )
        
        # Convert Markdown to HTML with basic styling
        html_content = markdown.markdown(markdown_content, extensions=['extra', 'tables'])
        
        # Add CSS for better PDF rendering with font fallbacks
        css = """
        <style>
            @font-face {
                font-family: 'Noto Sans CJK SC';
                src: local('Noto Sans CJK SC'), local('NotoSansCJKsc-Regular');
                font-weight: normal;
            }
            body {
                font-family: 'Noto Sans CJK SC', 'Source Han Sans CN', 'PingFang SC', 'Microsoft YaHei', sans-serif;
                margin: 20px;
                line-height: 1.6;
            }
            h1 { font-size: 24px; color: #333; }
            h2 { font-size: 20px; color: #444; }
            h3 { font-size: 18px; color: #555; }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 10px 0;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th { background-color: #f2f2f2; }
            img { max-width: 100%; height: auto; display: block; }
            pre, code { background-color: #f4f4f4; padding: 5px; }
        </style>
        """
        html_content = f"<html><head><meta charset='UTF-8'>{css}</head><body>{html_content}</body></html>"
        
        # Save temporary HTML file
        temp_html = markdown_file.replace('.md', '.temp.html')
        with open(temp_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Specify wkhtmltopdf path explicitly
        wkhtmltopdf_path = '/usr/bin/wkhtmltopdf'
        configuration = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
        
        # Convert HTML to PDF with local file access enabled
        options = {
            'encoding': 'UTF-8',
            'enable-local-file-access': None,  # Allow file:// URLs
            'quiet': '',  # Suppress wkhtmltopdf console output
            'dpi': '300',  # Improve rendering quality
            'no-outline': None,  # Disable outline for cleaner PDF
            'disable-smart-shrinking': None,  # Prevent font rendering issues
            'load-error-handling': 'ignore',  # Ignore missing image errors
            'load-media-error-handling': 'ignore'  # Ignore media errors
        }
        pdfkit.from_file(temp_html, pdf_file, configuration=configuration, options=options)
        
        # Clean up temporary HTML file
        os.remove(temp_html)
        
        print(f"PDF报告生成成功: {pdf_file}")
        return True
    except Exception as e:
        print(f"转换Markdown到PDF时出错: {e}")
        print("请检查以下内容：")
        print("1. 确保 wkhtmltopdf 已正确安装：")
        print("   wkhtmltopdf --version")
        print("2. 确保图像文件存在且路径正确：")
        print(f"   ls -l {os.path.dirname(markdown_file)}/data/*/*.png")
        print("3. 确保有权限访问文件：")
        print(f"   chmod -R u+rw {os.path.dirname(markdown_file)}")
        print("4. 确保 pdfkit 版本兼容：")
        print("   pip install --upgrade pdfkit")
        print("5. 确保 Noto Sans CJK SC 字体已安装：")
        print("   fc-list :lang=zh | grep Noto")
        print("   sudo apt-get install -y fonts-noto-cjk")
        print("6. 尝试以下命令重新安装 wkhtmltopdf：")
        print("   sudo apt-get update")
        print("   sudo apt-get install -y wkhtmltopdf")
        print("   或者手动安装：")
        print("   wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6.1-2/wkhtmltox_0.12.6.1-2.jammy_amd64.deb")
        print("   sudo dpkg -i wkhtmltox_0.12.6.1-2.jammy_amd64.deb")
        print("   sudo apt-get install -f")
        return False
    
def convert_markdown_to_pdf3(markdown_file, pdf_file, font_prop=None):
    """将Markdown文件转换为PDF文件，确保中文显示正确"""
    try:
        # Read Markdown content
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Convert relative image paths to absolute file:// paths
        base_path = os.path.dirname(os.path.abspath(markdown_file))
        markdown_content = re.sub(
            r'!\[(.*?)\]\((data/.*?)\)',
            lambda m: f'![{m.group(1)}](file://{base_path}/{m.group(2)})',
            markdown_content
        )
        
        # Convert Markdown to HTML with basic styling
        html_content = markdown.markdown(markdown_content, extensions=['extra', 'tables'])
        
        # Add CSS for better PDF rendering with font fallbacks
        css = """
        <style>
            @font-face {
                font-family: 'Noto Sans CJK SC';
                src: local('Noto Sans CJK SC'), local('NotoSansCJKsc-Regular');
                font-weight: normal;
            }
            body {
                font-family: 'Noto Sans CJK SC', 'Source Han Sans CN', 'PingFang SC', 'Microsoft YaHei', sans-serif;
                margin: 20px;
                line-height: 1.6;
            }
            h1 { font-size: 24px; color: #333; }
            h2 { font-size: 20px; color: #444; }
            h3 { font-size: 18px; color: #555; }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 10px 0;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th { background-color: #f2f2f2; }
            img { max-width: 100%; height: auto; }
            pre, code { background-color: #f4f4f4; padding: 5px; }
        </style>
        """
        html_content = f"<html><head><meta charset='UTF-8'>{css}</head><body>{html_content}</body></html>"
        
        # Save temporary HTML file
        temp_html = markdown_file.replace('.md', '.temp.html')
        with open(temp_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Specify wkhtmltopdf path explicitly
        wkhtmltopdf_path = '/usr/bin/wkhtmltopdf'
        configuration = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
        
        # Convert HTML to PDF with local file access enabled
        options = {
            'encoding': 'UTF-8',
            'enable-local-file-access': None,  # Allow file:// URLs
            'quiet': '',  # Suppress wkhtmltopdf console output
            'dpi': '300',  # Improve rendering quality
            'no-outline': None,  # Disable outline for cleaner PDF
            'disable-smart-shrinking': None  # Prevent font rendering issues
        }
        pdfkit.from_file(temp_html, pdf_file, configuration=configuration, options=options)
        
        # Clean up temporary HTML file
        os.remove(temp_html)
        
        print(f"PDF报告生成成功: {pdf_file}")
        return True
    except Exception as e:
        print(f"转换Markdown到PDF时出错: {e}")
        print("请检查以下内容：")
        print("1. 确保 wkhtmltopdf 已正确安装：")
        print("   wkhtmltopdf --version")
        print("2. 确保图像文件存在且路径正确：")
        print(f"   ls -l {os.path.dirname(markdown_file)}/data/*/*.png")
        print("3. 确保有权限访问文件：")
        print(f"   chmod -R u+rw {os.path.dirname(markdown_file)}")
        print("4. 确保 pdfkit 版本兼容：")
        print("   pip install --upgrade pdfkit")
        print("5. 确保 Noto Sans CJK SC 字体已安装：")
        print("   fc-list :lang=zh | grep Noto")
        print("   sudo apt-get install -y fonts-noto-cjk")
        print("6. 尝试以下命令重新安装 wkhtmltopdf：")
        print("   sudo apt-get update")
        print("   sudo apt-get install -y wkhtmltopdf")
        print("   或者手动安装：")
        print("   wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6.1-2/wkhtmltox_0.12.6.1-2.jammy_amd64.deb")
        print("   sudo dpkg -i wkhtmltox_0.12.6.1-2.jammy_amd64.deb")
        print("   sudo apt-get install -f")
        return False

def generate_comparison_report(folder1, folder2, csv_dir, output_file, config=None, font_prop=None):
    """生成两个时间段的性能对比报告并转换为PDF"""
    if config is None:
        config = load_config()
    
    folder1_path = os.path.join(csv_dir, folder1)
    folder2_path = os.path.join(csv_dir, folder2)
    common_csv_files = get_common_csv_files(folder1_path, folder2_path)
    
    if not common_csv_files:
        print("错误：两个文件夹中没有共同的CSV文件可供比较")
        return False
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"# 性能测试报告\n\n生成时间: {now}\n\n"
    
    report += "## 对比概述\n\n"
    report += f"本报告比较了以下两个时间段的性能数据：\n\n"
    report += f"- **较新数据**: {folder1}\n"
    report += f"- **较旧数据**: {folder2}\n\n"
    report += f"共分析了{len(common_csv_files)}个共同的性能指标。每个指标的详细对比分析如下。\n\n"
    
    report += "## 目录\n\n"
    for csv_file in common_csv_files:
        title = os.path.splitext(csv_file)[0]
        report += f"- [{title}](#{title.lower().replace(' ', '-').replace('_', '-')})\n"
    report += "\n"
    
    all_metrics_changes = {}
    
    for csv_file in common_csv_files:
        title = os.path.splitext(csv_file)[0]
        
        csv_path1 = os.path.join(folder1_path, csv_file)
        csv_path2 = os.path.join(folder2_path, csv_file)
        
        df1 = load_data(csv_path1)
        df2 = load_data(csv_path2)
        
        if df1 is None or df2 is None:
            report += f"## {title}\n\n无法加载数据。\n\n"
            continue
            
        analysis1 = analyze_data(df1)
        analysis2 = analyze_data(df2)
        
        if analysis1 is None or analysis2 is None:
            report += f"## {title}\n\n无法分析数据。\n\n"
            continue
            
        stats1 = analysis1["stats"]
        stats2 = analysis2["stats"]
        
        report += f"## {title}\n\n"
        
        time_range1 = f"{df1.iloc[0, 0].strftime('%Y-%m-%d %H:%M:%S')} 至 {df1.iloc[-1, 0].strftime('%Y-%m-%d %H:%M:%S')}"
        time_range2 = f"{df2.iloc[0, 0].strftime('%Y-%m-%d %H:%M:%S')} 至 {df2.iloc[-1, 0].strftime('%Y-%m-%d %H:%M:%S')}"
        
        report += f"### 数据概述\n\n"
        report += f"- **较新数据时间范围**: {time_range1}\n"
        report += f"- **较新数据点数**: {len(df1)}\n"
        report += f"- **较旧数据时间范围**: {time_range2}\n"
        report += f"- **较旧数据点数**: {len(df2)}\n\n"
        
        report += "### 统计数据对比\n\n"
        report += "|指标|较旧数据|较新数据|变化|变化百分比|\n"
        report += "|---|---|---|---|---:|\n"
        
        numeric_columns1 = set(df1.select_dtypes(include=[np.number]).columns)
        numeric_columns2 = set(df2.select_dtypes(include=[np.number]).columns)
        common_columns = list(numeric_columns1.intersection(numeric_columns2))
        
        metric_changes = {}
        
        for col in common_columns:
            if col in stats1 and col in stats2:
                mean_old = stats2[col]["mean"]
                mean_new = stats1[col]["mean"]
                mean_change = mean_new - mean_old
                mean_change_pct = calculate_change_percentage(mean_old, mean_new)
                
                change_symbol = "↑" if mean_change > 0 else "↓" if mean_change < 0 else ""
                change_pct_str = f"{abs(mean_change_pct):.2f}% {change_symbol}"
                if abs(mean_change_pct) > 20:
                    change_pct_str = f"**{change_pct_str}**"
                
                report += f"|{col}|{mean_old:.2f}|{mean_new:.2f}|{mean_change:.2f}|{change_pct_str}|\n"
                
                metric_changes[col] = {
                    "old_mean": mean_old,
                    "new_mean": mean_new,
                    "change": mean_change,
                    "change_pct": mean_change_pct
                }
        
        report += "\n"
        
        report += "### 可视化分析\n\n"
        trend_img_path1 = f"data/{folder1}/{title}_trend.png"
        trend_img_path2 = f"data/{folder2}/{title}_trend.png"
        
        report += f"#### 趋势对比\n\n"
        report += f"**较新数据趋势**:\n\n"
        report += f"![{title} 较新趋势分析]({trend_img_path1})\n\n"
        report += f"**较旧数据趋势**:\n\n"
        report += f"![{title} 较旧趋势分析]({trend_img_path2})\n\n"
        
        comparison_img_path = os.path.join(os.path.dirname(output_file), f"data/{folder1}/{title}_comparison.png")
        os.makedirs(os.path.dirname(comparison_img_path), exist_ok=True)
        create_comparison_chart(metric_changes, comparison_img_path, title, font_prop)
        
        report += f"#### 指标对比图\n\n"
        report += f"![{title} 新旧数据对比](data/{folder1}/{title}_comparison.png)\n\n"
        
        report += "### 性能对比结论\n\n"
        
        conclusions = []
        metric_type = "未知"
        for mtype, keywords in config["metric_keywords"].items():
            if any(keyword in title.lower() for keyword in keywords):
                metric_type = {
                    "cpu": "CPU/负载",
                    "memory": "内存",
                    "disk": "磁盘/存储",
                    "network": "网络"
                }.get(mtype, mtype.capitalize())
                break
        
        significant_changes = []
        for col, change_data in metric_changes.items():
            if abs(change_data["change_pct"]) > 10:
                change_direction = "增加" if change_data["change_pct"] > 0 else "减少"
                significant_changes.append((col, change_data["change_pct"], change_direction))
        
        if significant_changes:
            conclusions.append(f"**{metric_type}指标变化显著**：")
            for col, change_pct, direction in significant_changes:
                conclusions.append(f"- {col} {direction}了 {abs(change_pct):.2f}%")
            
            if metric_type == "CPU/负载":
                avg_change = sum(c[1] for c in significant_changes) / len(significant_changes)
                if avg_change > 0:
                    conclusions.append(f"\n整体CPU负载相比之前有所增加，可能是系统负载增加或性能下降的迹象。建议检查系统负载变化的原因。")
                else:
                    conclusions.append(f"\n整体CPU负载相比之前有所减少，系统性能可能有所改善。")
            elif metric_type == "内存":
                avg_change = sum(c[1] for c in significant_changes) / len(significant_changes)
                if avg_change > 0:
                    conclusions.append(f"\n内存使用率增加，可能需要关注内存泄漏问题或应用程序内存使用增长。")
                else:
                    conclusions.append(f"\n内存使用率降低，内存管理可能有所改善。")
            elif metric_type == "磁盘/存储":
                avg_change = sum(c[1] for c in significant_changes) / len(significant_changes)
                if avg_change > 0:
                    conclusions.append(f"\n磁盘I/O或使用率增加，可能需要关注存储性能或容量问题。")
                else:
                    conclusions.append(f"\n磁盘I/O或使用率降低，存储性能可能有所改善。")
            elif metric_type == "网络":
                avg_change = sum(c[1] for c in significant_changes) / len(significant_changes)
                if avg_change > 0:
                    conclusions.append(f"\n网络流量或连接数增加，需要关注网络带宽和连接管理。")
                else:
                    conclusions.append(f"\n网络流量或连接数减少，网络负载有所降低。")
        else:
            conclusions.append(f"**{metric_type}指标变化不显著**：各项指标相比之前变化较小，系统在该方面表现稳定。")
        
        report += "\n".join(conclusions) + "\n\n"
        
        all_metrics_changes[title] = {
            "type": metric_type,
            "changes": metric_changes
        }
    
    report += "## 总体结论\n\n"
    
    type_changes = {}
    for title, data in all_metrics_changes.items():
        metric_type = data["type"]
        if metric_type not in type_changes:
            type_changes[metric_type] = []
        changes = data["changes"]
        if changes:
            avg_change = sum(c["change_pct"] for c in changes.values()) / len(changes)
            type_changes[metric_type].append((title, avg_change))
    
    summary_img_path = os.path.join(os.path.dirname(output_file), f"data/{folder1}/summary_comparison.png")
    os.makedirs(os.path.dirname(summary_img_path), exist_ok=True)
    create_summary_chart(type_changes, summary_img_path, font_prop)
    
    report += "### 总体性能变化图表\n\n"
    report += f"![总体性能变化对比](data/{folder1}/summary_comparison.png)\n\n"
    
    conclusions = ["根据对比分析，系统性能变化总结如下：\n"]
    
    for metric_type, changes in type_changes.items():
        if not changes:
            continue
        avg_type_change = sum(c[1] for c in changes) / len(changes)
        
        if metric_type == "CPU/负载":
            if avg_type_change > 5:
                conclusions.append(f"1. **CPU/负载**：整体负载增加了 {avg_type_change:.2f}%，系统处理能力可能受到影响，建议关注CPU使用率上升的原因。")
            elif avg_type_change < -5:
                conclusions.append(f"1. **CPU/负载**：整体负载减少了 {abs(avg_type_change):.2f}%，系统处理能力有所改善。")
            else:
                conclusions.append(f"1. **CPU/负载**：整体变化不大（{avg_type_change:.2f}%），系统负载保持稳定。")
        elif metric_type == "内存":
            if avg_type_change > 5:
                conclusions.append(f"2. **内存使用**：内存使用增加了 {avg_type_change:.2f}%，可能需要关注内存管理和应用程序内存使用情况。")
            elif avg_type_change < -5:
                conclusions.append(f"2. **内存使用**：内存使用减少了 {abs(avg_type_change):.2f}%，内存管理有所改善。")
            else:
                conclusions.append(f"2. **内存使用**：整体变化不大（{avg_type_change:.2f}%），内存使用保持稳定。")
        elif metric_type == "磁盘/存储":
            if avg_type_change > 5:
                conclusions.append(f"3. **磁盘/存储**：磁盘使用或I/O增加了 {avg_type_change:.2f}%，需要关注存储性能和容量。")
            elif avg_type_change < -5:
                conclusions.append(f"3. **磁盘/存储**：磁盘使用或I/O减少了 {abs(avg_type_change):.2f}%，存储性能有所改善。")
            else:
                conclusions.append(f"3. **磁盘/存储**：整体变化不大（{avg_type_change:.2f}%），存储性能保持稳定。")
        elif metric_type == "网络":
            if avg_type_change > 5:
                conclusions.append(f"4. **网络性能**：网络流量或连接增加了 {avg_type_change:.2f}%，需要关注网络带宽和连接管理。")
            elif avg_type_change < -5:
                conclusions.append(f"4. **网络性能**：网络流量或连接减少了 {abs(avg_type_change):.2f}%，网络负载有所降低。")
            else:
                conclusions.append(f"4. **网络性能**：整体变化不大（{avg_type_change:.2f}%），网络性能保持稳定。")
        else:
            if abs(avg_type_change) > 5:
                change_direction = "增加" if avg_type_change > 0 else "减少"
                conclusions.append(f"- **{metric_type}**：整体{change_direction}了 {abs(avg_type_change):.2f}%。")
            else:
                conclusions.append(f"- **{metric_type}**：整体变化不大（{avg_type_change:.2f}%），保持稳定。")
    
    significant_changes_count = sum(1 for changes in type_changes.values() 
                                 for _, change in changes if abs(change) > 10)
    
    if significant_changes_count > 0:
        conclusions.append(f"\n**建议**：本次对比发现{significant_changes_count}个指标存在显著变化（>10%），需要关注这些变化较大的指标，分析变化原因，并采取相应的优化措施。")
    else:
        conclusions.append(f"\n**建议**：本次对比未发现显著变化的指标，系统整体性能保持稳定，建议继续保持当前的系统配置和监控。")
    
    report += "\n".join(conclusions) + "\n"
    
    # Write Markdown file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Convert Markdown to PDF
    pdf_file = output_file.replace('.md', '.pdf')
    convert_markdown_to_pdf(output_file, pdf_file, font_prop)
    
    return True

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="生成Grafana CSV数据的性能测试对比报告")
    parser.add_argument("--csv-dir", help="CSV文件目录路径", default=None)
    parser.add_argument("--output", help="输出的Markdown文件路径", default=None)
    parser.add_argument("--config", help="自定义配置文件路径", default=None)
    parser.add_argument("--folder1", help="较新的数据文件夹名称（可选，默认使用最新的）", default=None)
    parser.add_argument("--folder2", help="较旧的数据文件夹名称（可选，默认使用次新的）", default=None)
    args = parser.parse_args()
    
    # 设置中文字体
    font_found, font_prop = setup_chinese_font()
    if not font_found:
        print("错误：字体设置失败，退出程序")
        return 1
    
    # 设置输入和输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = args.csv_dir if args.csv_dir else os.path.join(script_dir, "grafana_csv")
    output_file = args.output if args.output else os.path.join(script_dir, "report.md")
    config_file = args.config if args.config else os.path.join(script_dir, "analysis_config.json")
    
    # 加载配置
    config = load_config(config_file)
    
    if not os.path.exists(csv_dir):
        print(f"错误：CSV目录不存在: {csv_dir}")
        return 1
    
    if args.folder1 and args.folder2:
        folder1 = args.folder1
        folder2 = args.folder2
    else:
        folders = get_comparison_folders(csv_dir)
        if not folders:
            print("错误：无法找到足够的日期文件夹进行比较")
            return 1
        folder1, folder2 = folders
    
    folder1_path = os.path.join(csv_dir, folder1)
    folder2_path = os.path.join(csv_dir, folder2)
    
    if not os.path.exists(folder1_path) or not os.path.exists(folder2_path):
        print(f"错误：指定的文件夹不存在: {folder1_path} 或 {folder2_path}")
        return 1
    
    print(f"正在生成性能测试对比报告...")
    print(f"使用配置文件: {config_file}")
    print(f"较新数据文件夹: {folder1}")
    print(f"较旧数据文件夹: {folder2}")
    print(f"输出报告文件: {output_file}")
    
    if generate_comparison_report(folder1, folder2, csv_dir, output_file, config, font_prop):
        print(f"对比报告生成成功: {output_file}")
    else:
        print(f"对比报告生成失败")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
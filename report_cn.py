#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用于从 Grafana 提取仪表板数据、分析 CSV 文件并生成性能比较报告。
python3 report.py --mode extract
python3 report.py --mode analyze
python3 report.py --mode report
"""

import os
import sys
import json
import time
import base64
import urllib.parse
import requests
import getpass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import markdown
import pdfkit
import re
import warnings
from datetime import datetime
from pathlib import Path
import argparse

# 忽略字体缺失警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph.*missing from current font.*")

DEFAULT_BASIC = {
    "grafana_url": "http://192.168.8.244:3000",
    "username": "admin",
    "password": "password",
    "dashboard": "VM-exporter",
    "time_range": "1h",
    "csv_dir": "./grafana_csv",
    "output_dir": "./data",
    "output_file": "./report.md"
}

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

# 指标类型关键字配置
METRIC_TYPE_KEYWORDS = {
    "cpu": ["cpu", "负载", "load"],
    "memory": ["内存", "memory", "ram"],
    "disk": ["磁盘", "disk", "iops", "io", "存储"],
    "network": ["网络", "network", "数据包", "packet", "流量", "traffic", "socket", "连接"]
}

# --- 字体设置 ---
def setup_chinese_font():
    """设置支持中文显示的字体为 SimSun。"""
    mpl.rcParams.update(mpl.rcParamsDefault)
    font_path = r"./simsun.ttc"
    
    if not os.path.exists(font_path):
        print(f"错误: 指定的字体文件不存在: {font_path}")
        return False, None
    
    try:
        fm.fontManager.addfont(font_path)
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['SimSun']
        plt.rcParams['axes.unicode_minus'] = False
        print(f"字体设置成功: SimSun（路径: {font_path}）")
        return True, font_prop
    except Exception as e:
        print(f"设置字体失败: {e}")
        return False, None

# --- Grafana 数据提取 ---
def get_grafana_api_key(grafana_url, username, password, output_dir):
    """使用服务账户 API 生成 Grafana API 密钥。"""
    api_key_file = os.path.join(output_dir, '.grafana_api_key')
    if os.path.exists(api_key_file):
        with open(api_key_file, 'r') as f:
            stored_data = json.load(f)
            if time.time() - stored_data.get('created_at', 0) < 30 * 24 * 60 * 60:
                print("使用已存储的 Grafana API 密钥")
                return stored_data['api_key']
    
    print(f"正在为 {username} 生成新的 Grafana API 密钥...")
    auth = base64.b64encode(f"{username}:{password}".encode()).decode("utf-8")
    headers = {
        "Authorization": f"Basic {auth}",
        "Content-Type": "application/json"
    }
    
    sa_name = f"csv_extractor_{int(time.time())}"
    sa_payload = {"name": sa_name, "role": "Viewer"}
    sa_response = requests.post(f"{grafana_url}/api/serviceaccounts", headers=headers, json=sa_payload)
    
    if sa_response.status_code not in [200, 201]:
        print(f"创建服务账户失败: 状态码 {sa_response.status_code}, 错误信息: {sa_response.text}")
        return None
    
    sa_id = sa_response.json().get('id')
    if not sa_id:
        print("无法获取服务账户 ID")
        return None
    
    token_payload = {"name": f"token_{int(time.time())}", "secondsToLive": 30 * 24 * 60 * 60}
    token_response = requests.post(f"{grafana_url}/api/serviceaccounts/{sa_id}/tokens", headers=headers, json=token_payload)
    
    if token_response.status_code not in [200, 201]:
        print(f"创建服务账户令牌失败: 状态码 {token_response.status_code}, 错误信息: {token_response.text}")
        return None
    
    api_key = token_response.json().get('key')
    if not api_key:
        print("无法获取服务账户令牌")
        return None
    
    with open(api_key_file, 'w') as f:
        json.dump({'api_key': api_key, 'created_at': time.time()}, f)
    
    print(f"成功为服务账户 {sa_name}（ID: {sa_id}）创建 API 密钥")
    return api_key

def find_dashboard_by_name(grafana_url, api_key, dashboard_name):
    """按名称查找仪表板。"""
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    search_url = f"{grafana_url}/api/search?query={urllib.parse.quote(dashboard_name)}"
    response = requests.get(search_url, headers=headers)
    
    if response.status_code == 200:
        dashboards = response.json()
        for dashboard in dashboards:
            if dashboard.get('title', '').lower() == dashboard_name.lower():
                print(f"找到仪表板: {dashboard.get('title')}（UID: {dashboard.get('uid')}）")
                return dashboard.get('uid')
        if dashboards:
            print(f"未找到完全匹配的仪表板, 使用: {dashboards[0].get('title')}（UID: {dashboards[0].get('uid')}）")
            return dashboards[0].get('uid')
    print(f"未找到仪表板: {dashboard_name}")
    return None

def get_dashboard_data(grafana_url, api_key, dashboard_uid):
    """获取仪表板数据。"""
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    dashboard_url = f"{grafana_url}/api/dashboards/uid/{dashboard_uid}"
    response = requests.get(dashboard_url, headers=headers)
    
    if response.status_code == 200:
        return response.json().get('dashboard', {})
    print(f"获取仪表板数据失败: 状态码 {response.status_code}")
    return None

def convert_grafana_time(time_str):
    """将 Grafana 时间字符串转换为 Unix 时间戳。"""
    if time_str == 'now':
        return int(time.time())
    if time_str.startswith('now-'):
        offset = time_str[4:]
        unit = offset[-1]
        value = int(offset[:-1])
        current_time = time.time()
        if unit == 'h':
            return int(current_time - value * 3600)
        elif unit == 'd':
            return int(current_time - value * 86400)
        elif unit == 'w':
            return int(current_time - value * 604800)
        elif unit == 'm':
            return int(current_time - value * 60)
    try:
        dt = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.%fZ')
        return int(dt.timestamp())
    except ValueError:
        try:
            dt = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%SZ')
            return int(dt.timestamp())
        except ValueError:
            print(f"无法解析时间字符串: {time_str}")
            return int(time.time() - 86400)

def extract_panel_csv(grafana_url, api_key, dashboard_data, panel, output_dir):
    """从面板提取 CSV 数据。"""
    try:
        panel_id = panel.get('id')
        panel_title = panel.get('title', f"panel_{panel_id}")
        safe_panel_title = "".join([c if c.isalnum() or c in "-_" else "_" for c in panel_title.replace(" ", "_")]).strip()
        
        time_range = dashboard_data.get('time', {'from': 'now-7d', 'to': 'now'})
        datasource = panel.get('datasource', {})
        datasource_uid = None
        
        if isinstance(datasource, dict):
            datasource_uid = datasource.get('uid')
        elif isinstance(datasource, str):
            datasource_uid = datasource
        else:
            for var in dashboard_data.get('templating', {}).get('list', []):
                if var.get('type') == 'datasource' and isinstance(var.get('current'), dict):
                    datasource_uid = var.get('current', {}).get('value')
                    break
        
        if not datasource_uid:
            print(f"警告: 面板 '{panel_title}'（ID: {panel_id}）无法确定数据源")
            return None
        
        template_vars = {}
        for var in dashboard_data.get('templating', {}).get('list', []):
            if var.get('type') != 'datasource':
                var_name = var.get('name', '')
                if var_name and isinstance(var.get('current'), dict):
                    var_value = var.get('current', {}).get('value')
                    if var_value is not None:
                        template_vars[var_name] = var_value
        
        if 'interval' not in template_vars:
            start_time = convert_grafana_time(time_range['from'])
            end_time = convert_grafana_time(time_range['to'])
            time_diff = end_time - start_time
            if time_diff > 86400 * 30:
                template_vars['interval'] = '1d'
            elif time_diff > 86400 * 7:
                template_vars['interval'] = '12h'
            elif time_diff > 86400:
                template_vars['interval'] = '1h'
            elif time_diff > 3600:
                template_vars['interval'] = '5m'
            else:
                template_vars['interval'] = '1m'
        
        queries = []
        for i, target in enumerate(panel.get('targets', [])):
            if target.get('hide'):
                continue
            expr = target.get('expr') or target.get('query')
            if not expr:
                continue
            for var_name, var_value in template_vars.items():
                expr = expr.replace(f"${var_name}", str(var_value))
                expr = expr.replace(f"${{{var_name}}}", str(var_value))
            queries.append({'expr': expr, 'legend': target.get('legendFormat', f"series_{i}")})
        
        if not queries:
            print(f"警告: 面板 '{panel_title}'（ID: {panel_id}）没有有效的查询")
            return None
        
        all_data_frames = []
        for i, query in enumerate(queries):
            csv_url = f"{grafana_url}/api/datasources/proxy/uid/{datasource_uid}/api/v1/query_range"
            start_time = convert_grafana_time(time_range['from'])
            end_time = convert_grafana_time(time_range['to'])
            time_diff = end_time - start_time
            step = '1h' if time_diff > 86400 * 30 else '15m' if time_diff > 86400 * 7 else '5m' if time_diff > 86400 else '60s'
            
            params = {'query': query['expr'], 'start': start_time, 'end': end_time, 'step': step}
            headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
            
            print(f"正在为面板 '{panel_title}'（ID: {panel_id}）获取查询 {i+1}/{len(queries)} 的数据...")
            response = requests.get(csv_url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success':
                    results = data['data']['result']
                    if not results:
                        print(f"查询 {i+1} 未返回数据点")
                        continue
                    for result_idx, result in enumerate(results):
                        metric = result.get('metric', {})
                        legend = query['legend']
                        for key, value in metric.items():
                            legend = legend.replace(f"{{{{{key}}}}}", str(value))
                        if legend in ["", "{}"]:
                            legend = "_".join([f"{k}={v}" for k, v in metric.items() if k != "__name__"]) or f"series_{i}_{result_idx}"
                        counter = 1
                        original_legend = legend
                        while any(legend in df.columns for df in all_data_frames):
                            legend = f"{original_legend}_{counter}"
                            counter += 1
                        values = result.get('values', [])
                        if values:
                            df = pd.DataFrame(values, columns=['timestamp', legend])
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                            df[legend] = df[legend].astype(float)
                            df.set_index('timestamp', inplace=True)
                            all_data_frames.append(df)
            else:
                print(f"获取面板 '{panel_title}'（ID: {panel_id}）数据失败: 状态码 {response.status_code}")
        
        if all_data_frames:
            merged_df = pd.concat(all_data_frames, axis=1)
            current_time = datetime.now()
            date_folder = current_time.strftime("%Y-%m-%d-%H:%M")
            csv_folder = os.path.join(output_dir, "grafana_csv", date_folder)
            os.makedirs(csv_folder, exist_ok=True)
            filename = f"{safe_panel_title}_panel_{panel_id}.csv"
            filepath = os.path.join(csv_folder, filename)
            merged_df.to_csv(filepath)
            print(f"已将面板 '{panel_title}'（ID: {panel_id}）数据保存到: {filepath}")
            return filepath
        print(f"面板 '{panel_title}'（ID: {panel_id}）未返回有效数据")
        return None
    except Exception as e:
        print(f"提取面板 CSV 数据时出错: {str(e)}")
        return None

def extract_all_panel_csvs(grafana_url, api_key, dashboard_name, time_range, output_dir, to_time=None):
    """为仪表板中的所有面板提取 CSV 数据。
    
    参数:
        grafana_url: Grafana 服务器 URL
        api_key: Grafana API 密钥
        dashboard_name: 仪表板名称
        time_range: 时间范围，例如 "1h"、"7d" 或 "2023-01-01,2023-01-31"
        output_dir: 输出目录
        to_time: 可选的结束时间，相对于当前时间的偏移，例如 "1h"、"2d"，默认为 None（即当前时间）
    """
    dashboard_uid = find_dashboard_by_name(grafana_url, api_key, dashboard_name)
    if not dashboard_uid:
        return []
    
    dashboard_data = get_dashboard_data(grafana_url, api_key, dashboard_uid)
    if not dashboard_data:
        return []
    
    if time_range:
        if "," in time_range:
            from_time, to_time_str = time_range.split(",")
            dashboard_data['time'] = {'from': from_time.strip(), 'to': to_time_str.strip()}
        else:
            # 处理结束时间
            to_time_str = 'now'
            if to_time:
                # 如果指定了结束时间偏移，则计算相对于当前时间的结束时间点
                to_time_str = f'now-{to_time}'
            
            dashboard_data['time'] = {'from': f'now-{time_range}', 'to': to_time_str}
        print(f"使用自定义时间范围: {dashboard_data['time']['from']} 到 {dashboard_data['time']['to']}")
    
    panels = dashboard_data.get('panels', [])
    for panel in list(panels):
        if panel.get('type') == 'row':
            panels.extend(panel.get('panels', []))
    
    csv_files = []
    for panel in panels:
        if panel.get('type') in ['row', 'text', 'dashlist', 'news']:
            continue
        csv_file = extract_panel_csv(grafana_url, api_key, dashboard_data, panel, output_dir)
        if csv_file:
            csv_files.append(csv_file)
    
    return csv_files

# --- 数据分析 ---
def load_data(file_path):
    """将 CSV 数据加载到 pandas DataFrame 中。"""
    try:
        df = pd.read_csv(file_path)
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def analyze_data(df):
    """分析 DataFrame 中的数值列。"""
    try:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        stats = {
            col: {
                "min": df[col].min(),
                "max": df[col].max(),
                "mean": df[col].mean(),
                "median": df[col].median(),
                "std": df[col].std()
            } for col in numeric_columns
        }
        return {"df": df, "stats": stats}
    except Exception as e:
        print(f"分析数据失败: {e}")
        return None

def plot_trend(analysis_result, output_path, title, font_prop=None):
    """为数值列绘制趋势分析图。"""
    df = analysis_result["df"]
    stats = analysis_result["stats"]
    plt.figure(figsize=(14, 10))
    time_col = df.columns[0]
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        plt.plot(df[time_col], df[col], 'o-', markersize=3, linewidth=1, alpha=0.7, label=col)
        plt.axhline(y=stats[col]["mean"], linestyle='--', alpha=0.5, label=f'{col} 均值: {stats[col]["mean"]:.2f}')
    
    plt.title(f'{title} - 趋势分析', fontsize=16, pad=20, fontproperties=font_prop)
    plt.xlabel('时间', fontsize=12, fontproperties=font_prop)
    plt.ylabel('值', fontsize=12, fontproperties=font_prop)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', prop=font_prop)
    plt.xticks(rotation=45)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
    try:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"趋势图已保存到: {output_path}")
    except Exception as e:
        print(f"保存趋势图失败: {e}")
    finally:
        plt.close()

def plot_histogram(analysis_result, output_path, title, font_prop=None):
    """为数值列绘制直方图。"""
    df = analysis_result["df"]
    stats = analysis_result["stats"]
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    num_cols = len(numeric_columns)
    
    if num_cols == 0:
        return
    
    fig, axes = plt.subplots(1, num_cols, figsize=(6*num_cols, 8))
    if num_cols == 1:
        axes = [axes]
    
    for i, col in enumerate(numeric_columns):
        data = df[col].dropna().values
        if len(data) == 0:
            axes[i].text(0.5, 0.5, "无有效数据", ha='center', va='center', fontproperties=font_prop)
        elif len(np.unique(data)) <= 1:
            axes[i].axvline(data[0], color='b', linewidth=2)
            axes[i].text(data[0], 0.5, f"固定值: {data[0]}", ha='center', va='center', fontproperties=font_prop)
        else:
            try:
                min_val, max_val = np.min(data), np.max(data)
                bins = [min_val - 0.5, min_val + 0.5] if np.isclose(min_val, max_val) else np.linspace(min_val, max_val, 31)
                axes[i].hist(data, bins=bins, alpha=0.7)
            except Exception as e:
                print(f"绘制直方图失败（{col}）: {e}")
                axes[i].text(0.5, 0.5, f"无法绘制直方图: {str(e)}", ha='center', va='center', fontproperties=font_prop)
        
        if col in stats:
            if "mean" in stats[col] and not np.isnan(stats[col]["mean"]):
                axes[i].axvline(stats[col]["mean"], color='r', linestyle='--', label=f'均值: {stats[col]["mean"]:.2f}')
            if "median" in stats[col] and not np.isnan(stats[col]["median"]):
                axes[i].axvline(stats[col]["median"], color='g', linestyle='--', label=f'中位数: {stats[col]["median"]:.2f}')
        
        axes[i].set_title(f'{col} 分布', fontproperties=font_prop)
        axes[i].set_xlabel('值', fontproperties=font_prop)
        axes[i].set_ylabel('频率', fontproperties=font_prop)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(prop=font_prop)
    
    plt.suptitle(f'{title} - 分布分析', fontsize=16, y=0.98, fontproperties=font_prop)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.15, wspace=0.3)
    try:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"直方图已保存到: {output_path}")
    except Exception as e:
        print(f"保存直方图失败: {e}")
    finally:
        plt.close()

def analyze_csv_file(csv_path, output_dir, font_prop=None):
    """分析单个 CSV 文件并生成趋势图和直方图。"""
    title = os.path.splitext(os.path.basename(csv_path))[0]
    safe_title = title.replace(' ', '_')
    df = load_data(csv_path)
    if df is None:
        return False
    
    analysis_result = analyze_data(df)
    if analysis_result is None:
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    trend_path = os.path.join(output_dir, f"{safe_title}_trend.png")
    plot_trend(analysis_result, trend_path, title, font_prop)
    hist_path = os.path.join(output_dir, f"{safe_title}_histogram.png")
    plot_histogram(analysis_result, hist_path, title, font_prop)
    return True

def analyze_csv_directory(csv_dir, output_dir, font_prop=None):
    """分析目录结构中的所有 CSV 文件。"""
    if not os.path.exists(csv_dir):
        print(f"错误: CSV 目录不存在: {csv_dir}")
        return 1
    
    dir_items = os.listdir(csv_dir)
    date_dirs = [d for d in dir_items if os.path.isdir(os.path.join(csv_dir, d))]
    
    total_files = 0
    success_count = 0
    
    if date_dirs:
        print(f"找到 {len(date_dirs)} 个日期文件夹")
        for date_dir in date_dirs:
            date_csv_dir = os.path.join(csv_dir, date_dir)
            date_output_dir = os.path.join(output_dir, date_dir)
            if os.path.exists(date_output_dir) and os.listdir(date_output_dir):
                print(f"跳过已处理的日期文件夹: {date_dir}")
                continue
            
            csv_files = [f for f in os.listdir(date_csv_dir) if f.endswith('.csv')]
            if not csv_files:
                print(f"警告: 在 {date_csv_dir} 中未找到 CSV 文件")
                continue
            
            print(f"正在处理日期文件夹: {date_dir}（共 {len(csv_files)} 个 CSV 文件）")
            dir_success_count = 0
            for csv_file in csv_files:
                csv_path = os.path.join(date_csv_dir, csv_file)
                print(f"  正在分析: {csv_file}")
                if analyze_csv_file(csv_path, date_output_dir, font_prop):
                    dir_success_count += 1
                    success_count += 1
            total_files += len(csv_files)
            print(f"  完成对 {date_dir} 的分析: 处理了 {dir_success_count}/{len(csv_files)} 个文件")
    else:
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"错误: 在 {csv_dir} 中未找到 CSV 文件")
            return 1
        for csv_file in csv_files:
            csv_path = os.path.join(csv_dir, csv_file)
            print(f"正在分析: {csv_file}")
            if analyze_csv_file(csv_path, output_dir, font_prop):
                success_count += 1
        total_files += len(csv_files)
    
    print(f"\n分析完成: 成功处理了 {success_count}/{total_files} 个 CSV 文件")
    print(f"结果已保存到: {output_dir}")
    return 0

# --- 报告生成 ---
def load_config(config_path=None):
    """加载自定义配置或使用默认配置。"""
    config = {"basic":DEFAULT_BASIC.copy(),"thresholds": DEFAULT_THRESHOLDS.copy(), "metric_keywords": METRIC_TYPE_KEYWORDS.copy()}
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            if "basic" in user_config:
                config["basic"].update(user_config["basic"])
            if "thresholds" in user_config:
                config["thresholds"].update(user_config["thresholds"])
            if "metric_keywords" in user_config:
                for metric_type, keywords in user_config["metric_keywords"].items():
                    config["metric_keywords"][metric_type] = list(set(
                        config["metric_keywords"].get(metric_type, []) + keywords
                    ))
            print(f"已加载自定义配置文件: {config_path}")
        except Exception as e:
            print(f"加载配置文件失败: {e}, 使用默认配置")
    return config

def find_date_folders(base_dir):
    """查找日期格式的文件夹并按日期排序。"""
    if not os.path.exists(base_dir) or not os.path.isdir(base_dir):
        print(f"错误: 目录不存在: {base_dir}")
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
                print(f"警告: 文件夹名称 {item} 符合日期格式但无法解析")
    date_folders.sort(key=lambda x: x[1], reverse=True)
    return [folder[0] for folder in date_folders]

def get_comparison_folders(base_dir):
    """获取用于比较的两个最新文件夹。"""
    date_folders = find_date_folders(base_dir)
    if len(date_folders) < 2:
        print(f"错误: 需要至少两个日期文件夹进行比较, 仅找到 {len(date_folders)} 个")
        return None
    return date_folders[0], date_folders[1]

def get_common_csv_files(folder1_path, folder2_path):
    """获取两个文件夹中共同的 CSV 文件。"""
    if not os.path.exists(folder1_path) or not os.path.exists(folder2_path):
        return []
    files1 = set([f for f in os.listdir(folder1_path) if f.endswith('.csv')])
    files2 = set([f for f in os.listdir(folder2_path) if f.endswith('.csv')])
    return list(files1.intersection(files2))

def calculate_change_percentage(value1, value2):
    """计算两个值之间的百分比变化。"""
    if value1 == 0:
        return float('inf') if value2 > 0 else 0.0
    return ((value2 - value1) / abs(value1)) * 100.0

def create_comparison_chart(metric_changes, output_path, title, font_prop=None):
    """创建比较新旧数据的柱状图。"""
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
    plt.bar(x - width/2, old_values, width, label='旧数据', alpha=0.7, color='#1f77b4')
    plt.bar(x + width/2, new_values, width, label='新数据', alpha=0.7, color='#ff7f0e')
    
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
    plt.ylabel('值', fontsize=12, fontproperties=font_prop)
    plt.title(f'{title} - 数据比较', fontsize=14, fontproperties=font_prop)
    plt.xticks(x, metrics, rotation=45, ha='right', fontproperties=font_prop)
    plt.legend(prop=font_prop)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

def create_summary_chart(type_changes, output_path, font_prop=None):
    """创建性能变化的汇总图表。"""
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
    plt.title('按指标类型的系统性能变化', fontsize=14, fontproperties=font_prop)
    plt.xticks(x, types, fontproperties=font_prop)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(prop=font_prop)
    plt.ylim(min(plt.ylim()[0], -10), max(plt.ylim()[1], 15))
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

def convert_markdown_to_pdf(markdown_file, pdf_file, font_prop=None):
    """将 Markdown 文件转换为 PDF, 支持中文字体。"""
    try:
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        base_path = os.path.dirname(os.path.abspath(markdown_file))
        def replace_image_path(match):
            alt_text = match.group(1)
            rel_path = match.group(2)
            abs_path = os.path.normpath(os.path.join(base_path, rel_path))
            if not os.path.exists(abs_path):
                print(f"警告: 图像文件不存在: {abs_path}")
                return f'![{alt_text}](about:blank)'
            file_url = 'file:///' + urllib.parse.quote(abs_path.replace(os.sep, '/'), safe='/')
            return f'![{alt_text}]({file_url})'
        
        markdown_content = re.sub(r'!\[(.*?)\]\(([^)]+)\)', replace_image_path, markdown_content, flags=re.UNICODE)
        html_content = markdown.markdown(markdown_content, extensions=['extra', 'tables'])
        
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
        
        temp_html = markdown_file.replace('.md', '.temp.html')
        with open(temp_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        wkhtmltopdf_path = '/usr/bin/wkhtmltopdf'
        configuration = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
        options = {
            'encoding': 'UTF-8',
            'enable-local-file-access': None,
            'quiet': '',
            'dpi': '300',
            'no-outline': None,
            'disable-smart-shrinking': None,
            'load-error-handling': 'ignore',
            'load-media-error-handling': 'ignore'
        }
        pdfkit.from_file(temp_html, pdf_file, configuration=configuration, options=options)
        os.remove(temp_html)
        print(f"PDF 报告生成成功: {pdf_file}")
        return True
    except Exception as e:
        print(f"将 Markdown 转换为 PDF 失败: {e}")
        return False

def generate_comparison_report(folder1, folder2, csv_dir, output_file, config=None, font_prop=None):
    """生成性能比较报告并转换为 PDF。"""
    if config is None:
        config = load_config()
    
    folder1_path = os.path.join(csv_dir, folder1)
    folder2_path = os.path.join(csv_dir, folder2)
    common_csv_files = get_common_csv_files(folder1_path, folder2_path)
    
    if not common_csv_files:
        print("错误: 未找到可比较的共同 CSV 文件")
        return False
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"# 性能测试报告\n\n生成时间: {now}\n\n"
    report += "## 比较概览\n\n"
    report += f"本报告比较了两个时间段的性能数据: \n\n"
    report += f"- **新数据**: {folder1}\n"
    report += f"- **旧数据**: {folder2}\n\n"
    report += f"分析了 {len(common_csv_files)} 个共同的性能指标。\n\n"
    
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
        
        time_range1 = f"{df1.iloc[0, 0].strftime('%Y-%m-%d %H:%M:%S')} 到 {df1.iloc[-1, 0].strftime('%Y-%m-%d %H:%M:%S')}"
        time_range2 = f"{df2.iloc[0, 0].strftime('%Y-%m-%d %H:%M:%S')} 到 {df2.iloc[-1, 0].strftime('%Y-%m-%d %H:%M:%S')}"
        report += f"### 数据概览\n\n"
        report += f"- **新数据时间范围**: {time_range1}\n"
        report += f"- **新数据点数**: {len(df1)}\n"
        report += f"- **旧数据时间范围**: {time_range2}\n"
        report += f"- **旧数据点数**: {len(df2)}\n\n"
        
        report += "### 统计比较\n\n"
        report += "|指标|旧数据|新数据|变化|变化百分比|\n"
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
        report += f"#### 趋势比较\n\n"
        report += f"**新数据趋势**: \n\n"
        report += f"![{title} 新数据趋势分析]({trend_img_path1})\n\n"
        report += f"**旧数据趋势**: \n\n"
        report += f"![{title} 旧数据趋势分析]({trend_img_path2})\n\n"
        
        comparison_img_path = os.path.join(os.path.dirname(output_file), f"data/{folder1}/{title}_comparison.png")
        os.makedirs(os.path.dirname(comparison_img_path), exist_ok=True)
        create_comparison_chart(metric_changes, comparison_img_path, title, font_prop)
        report += f"#### 指标比较图表\n\n"
        report += f"![{title} 数据比较](data/{folder1}/{title}_comparison.png)\n\n"
        
        report += "### 性能比较结论\n\n"
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
        
        significant_changes = [(col, change_data["change_pct"], "增加" if change_data["change_pct"] > 0 else "减少")
                              for col, change_data in metric_changes.items() if abs(change_data["change_pct"]) > 10]
        
        if significant_changes:
            conclusions.append(f"**{metric_type} 指标的显著变化**: ")
            for col, change_pct, direction in significant_changes:
                conclusions.append(f"- {col} {direction}了 {abs(change_pct):.2f}%")
            if metric_type == "CPU/负载":
                avg_change = sum(c[1] for c in significant_changes) / len(significant_changes)
                conclusions.append(f"\n总体 CPU 负载{'增加' if avg_change > 0 else '减少'}, "
                                  f"{'可能表明负载增加或性能下降, 请调查原因。' if avg_change > 0 else '表明性能可能改善。'}")
            elif metric_type == "内存":
                avg_change = sum(c[1] for c in significant_changes) / len(significant_changes)
                conclusions.append(f"\n内存使用量{'增加, 请检查内存泄漏或应用使用量增加。' if avg_change > 0 else '减少, 表明内存管理改善。'}")
            elif metric_type == "磁盘/存储":
                avg_change = sum(c[1] for c in significant_changes) / len(significant_changes)
                conclusions.append(f"\n磁盘 I/O 或使用量{'增加, 请监控存储性能或容量。' if avg_change > 0 else '减少, 表明存储性能改善。'}")
            elif metric_type == "网络":
                avg_change = sum(c[1] for c in significant_changes) / len(significant_changes)
                conclusions.append(f"\n网络流量或连接数{'增加, 请监控带宽和连接管理。' if avg_change > 0 else '减少, 表明网络负载降低。'}")
        else:
            conclusions.append(f"**{metric_type} 指标无显著变化**: 指标变化较小, 表明性能稳定。")
        
        report += "\n".join(conclusions) + "\n\n"
        all_metrics_changes[title] = {"type": metric_type, "changes": metric_changes}
    
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
    report += f"![总体性能比较](data/{folder1}/summary_comparison.png)\n\n"
    
    conclusions = ["根据比较结果, 系统性能变化总结如下: \n"]
    for metric_type, changes in type_changes.items():
        if not changes:
            continue
        avg_type_change = sum(c[1] for c in changes) / len(changes)
        if metric_type == "CPU/负载":
            conclusions.append(f"1. **CPU/负载**: {'增加' if avg_type_change > 5 else '减少' if avg_type_change < -5 else '稳定'} "
                              f"{abs(avg_type_change):.2f}%, {'请调查原因。' if avg_type_change > 5 else '表明性能改善。' if avg_type_change < -5 else '系统负载稳定。'}")
        elif metric_type == "内存":
            conclusions.append(f"2. **内存**: {'增加' if avg_type_change > 5 else '减少' if avg_type_change < -5 else '稳定'} "
                              f"{abs(avg_type_change):.2f}%, {'请检查内存管理。' if avg_type_change > 5 else '内存管理改善。' if avg_type_change < -5 else '内存使用稳定。'}")
        elif metric_type == "磁盘/存储":
            conclusions.append(f"3. **磁盘/存储**: {'增加' if avg_type_change > 5 else '减少' if avg_type_change < -5 else '稳定'} "
                              f"{abs(avg_type_change):.2f}%, {'请监控性能。' if avg_type_change > 5 else '存储性能改善。' if avg_type_change < -5 else '存储性能稳定。'}")
        elif metric_type == "网络":
            conclusions.append(f"4. **网络**: {'增加' if avg_type_change > 5 else '减少' if avg_type_change < -5 else '稳定'} "
                              f"{abs(avg_type_change):.2f}%, {'请监控带宽。' if avg_type_change > 5 else '网络负载降低。' if avg_type_change < -5 else '网络性能稳定。'}")
        else:
            conclusions.append(f"- **{metric_type}**: {'增加' if avg_type_change > 5 else '减少' if avg_type_change < -5 else '稳定'} "
                              f"{abs(avg_type_change):.2f}%。")
    
    significant_changes_count = sum(1 for changes in type_changes.values() for _, change in changes if abs(change) > 10)
    conclusions.append(f"\n**建议**: {'发现' if significant_changes_count > 0 else '未发现'}显著变化 "
                      f"({'请调查原因并优化。' if significant_changes_count > 0 else '维持当前配置。'})")
    
    report += "\n".join(conclusions) + "\n"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    pdf_file = output_file.replace('.md', '.pdf')
    convert_markdown_to_pdf(output_file, pdf_file, font_prop)
    return True

# --- 主函数 ---
def main():
    """主函数, 执行合并的工作流程。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_default = os.path.join(script_dir, "analysis_config.json")
    
    # 首先加载配置文件
    if os.path.exists(config_file_default):
        config = load_config(config_file_default)
        default_config = config.get("basic", {})
    else:
        default_config = DEFAULT_BASIC.copy()
    
    # 现在定义参数解析器, 不使用默认值
    parser = argparse.ArgumentParser(description="合并脚本, 用于 Grafana 数据提取、分析和报告生成。")
    parser.add_argument("--mode", choices=['all', 'extract', 'analyze', 'report'], 
                        help="执行模式: all（默认）、extract、analyze 或 report")
    parser.add_argument("--grafana-url", help="Grafana URL")
    parser.add_argument("--username", help="Grafana 用户名")
    parser.add_argument("--password", help="Grafana 密码")
    parser.add_argument("--dashboard", help="Grafana 仪表板名称")
    parser.add_argument("--time-range", help="时间范围")
    parser.add_argument("--to-time", help="结束时间，相对于当前时间的偏移，例如 1h、2d 等，默认为当前时间")
    parser.add_argument("--csv-dir", help="CSV 文件目录")
    parser.add_argument("--output-dir", help="分析结果输出目录")
    parser.add_argument("--output-file", help="报告输出的 Markdown 文件")
    parser.add_argument("--config", help="自定义配置文件")
    parser.add_argument("--folder1", help="新数据文件夹")
    parser.add_argument("--folder2", help="旧数据文件夹")
    
    args = parser.parse_args()
    
    # 如果指定了自定义配置文件, 重新加载配置
    if args.config:
        config = load_config(args.config)
        default_config = config.get("basic", {})
    
    # 使用参数或配置值
    mode = args.mode or 'all'
    grafana_url = args.grafana_url or default_config.get("grafana_url")
    username = args.username or default_config.get("username")
    password = args.password or default_config.get("password")
    dashboard_name = args.dashboard or default_config.get("dashboard")
    time_range = args.time_range or default_config.get("time_range")
    
    csv_dir = args.csv_dir or os.path.join(script_dir, default_config.get("csv_dir", "grafana_csv").lstrip("./"))
    output_dir = args.output_dir or os.path.join(script_dir, default_config.get("output_dir", "data").lstrip("./"))
    output_file = args.output_file or os.path.join(script_dir, default_config.get("output_file", "report.md").lstrip("./"))
    
    font_found, font_prop = setup_chinese_font()
    if not font_found:
        print("错误: 字体设置失败, 退出程序")
        return 1
    
    print(f"使用配置: grafana_url={grafana_url}, username={username}, dashboard={dashboard_name}")
    
    if args.mode in ['all', 'extract']:
        use_default = input(f"使用默认凭据 ({username}@{grafana_url})？(Y/n): ").lower() != "n"
        
        if not use_default:
            grafana_url = input(f"输入 Grafana URL（默认: {grafana_url}）: ") or grafana_url
            username = input(f"输入 Grafana 用户名（默认: {username}）: ") or username
            password = getpass.getpass("输入 Grafana 密码: ")
        
        # if not time_range:
        use_custom_time = input("使用自定义时间范围？(y/N): ").lower() == "y"
        if use_custom_time:
            time_range = input("输入时间范围（默认: 1h）: ") or "1h"
        
        api_key = get_grafana_api_key(grafana_url, username, password, script_dir)
        if not api_key:
            print("获取 Grafana API 密钥失败, 退出程序")
            return 1
        
        # 获取结束时间参数
        to_time = args.to_time
        
        start_time = time.time()
        csv_files = extract_all_panel_csvs(grafana_url, api_key, dashboard_name, time_range, script_dir, to_time)
        end_time = time.time()
        
        if csv_files:
            print("\n成功导出的 CSV 文件: ")
            for file in csv_files:
                print(f"- {file}")
            print(f"\n共导出 {len(csv_files)} 个面板, 耗时 {end_time - start_time:.2f} 秒")
        else:
            print("未能导出任何 CSV 文件")
            if args.mode == 'extract':
                return 1
    
    if args.mode in ['all', 'analyze']:
        result = analyze_csv_directory(csv_dir, output_dir, font_prop)
        if result != 0 and args.mode == 'analyze':
            return result
    
    if args.mode in ['all', 'report']:
        folder1 = args.folder1
        folder2 = args.folder2
        if not folder1 or not folder2:
            folders = get_comparison_folders(csv_dir)
            if not folders:
                print("错误: 可比较的日期文件夹不足")
                return 1
            folder1, folder2 = folders
        
        folder1_path = os.path.join(csv_dir, folder1)
        folder2_path = os.path.join(csv_dir, folder2)
        if not os.path.exists(folder1_path) or not os.path.exists(folder2_path):
            print(f"错误: 指定的文件夹不存在: {folder1_path} 或 {folder2_path}")
            return 1
        
        print(f"正在生成性能比较报告...")
        print(f"使用配置文件: {args.config}")
        print(f"新数据文件夹: {folder1}")
        print(f"旧数据文件夹: {folder2}")
        print(f"输出报告文件: {output_file}")
        
        if generate_comparison_report(folder1, folder2, csv_dir, output_file, config, font_prop):
            print(f"性能比较报告生成成功: {output_file}")
        else:
            print("生成性能比较报告失败")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

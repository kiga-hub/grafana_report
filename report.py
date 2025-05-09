#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for extracting Grafana dashboard data, analyzing CSV files, and generating performance comparison reports.
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

# Suppress font missing warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph.*missing from current font.*")

# 
DEFAULT_BASIC = {
    "grafana_url": "http://192.168.8.200:3000",
    "username": "admin",
    "password": "password",
    "dashboard": "VM-exporter",
    "time_range": "1h",
    "csv_dir": "./grafana_csv",
    "output_dir": "./data",
    "output_file": "./report.md"
}

# Default analysis thresholds configuration
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

# Metric type keywords configuration
METRIC_TYPE_KEYWORDS = {
    "cpu": ["cpu", "负载", "load"],
    "memory": ["内存", "memory", "ram"],
    "disk": ["磁盘", "disk", "iops", "io", "存储"],
    "network": ["网络", "network", "数据包", "packet", "流量", "traffic", "socket", "连接"]
}

# --- Font Setup ---
def setup_chinese_font():
    """Set up font to support Chinese display using SimSun."""
    mpl.rcParams.update(mpl.rcParamsDefault)
    font_path = r"/root/.fonts/simsun.ttc"
    
    if not os.path.exists(font_path):
        print(f"Error: Specified font file does not exist: {font_path}")
        return False, None
    
    try:
        fm.fontManager.addfont(font_path)
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['SimSun']
        plt.rcParams['axes.unicode_minus'] = False
        print(f"Font set: SimSun (path: {font_path})")
        return True, font_prop
    except Exception as e:
        print(f"Error setting font: {e}")
        return False, None

# --- Grafana Data Extraction ---
def get_grafana_api_key(grafana_url, username, password, output_dir):
    """Generate Grafana API key using Service Account API."""
    api_key_file = os.path.join(output_dir, '.grafana_api_key')
    if os.path.exists(api_key_file):
        with open(api_key_file, 'r') as f:
            stored_data = json.load(f)
            if time.time() - stored_data.get('created_at', 0) < 30 * 24 * 60 * 60:
                print("Using stored Grafana API key")
                return stored_data['api_key']
    
    print(f"Generating new Grafana API key for {username}...")
    auth = base64.b64encode(f"{username}:{password}".encode()).decode("utf-8")
    headers = {
        "Authorization": f"Basic {auth}",
        "Content-Type": "application/json"
    }
    
    sa_name = f"csv_extractor_{int(time.time())}"
    sa_payload = {"name": sa_name, "role": "Viewer"}
    sa_response = requests.post(f"{grafana_url}/api/serviceaccounts", headers=headers, json=sa_payload)
    
    if sa_response.status_code not in [200, 201]:
        print(f"Failed to create service account: {sa_response.status_code}, {sa_response.text}")
        return None
    
    sa_id = sa_response.json().get('id')
    if not sa_id:
        print("Failed to get service account ID")
        return None
    
    token_payload = {"name": f"token_{int(time.time())}", "secondsToLive": 30 * 24 * 60 * 60}
    token_response = requests.post(f"{grafana_url}/api/serviceaccounts/{sa_id}/tokens", headers=headers, json=token_payload)
    
    if token_response.status_code not in [200, 201]:
        print(f"Failed to create service account token: {token_response.status_code}, {token_response.text}")
        return None
    
    api_key = token_response.json().get('key')
    if not api_key:
        print("Failed to get service account token")
        return None
    
    with open(api_key_file, 'w') as f:
        json.dump({'api_key': api_key, 'created_at': time.time()}, f)
    
    print(f"Successfully created API key for service account {sa_name} (ID: {sa_id})")
    return api_key

def find_dashboard_by_name(grafana_url, api_key, dashboard_name):
    """Find dashboard by name."""
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    search_url = f"{grafana_url}/api/search?query={urllib.parse.quote(dashboard_name)}"
    response = requests.get(search_url, headers=headers)
    
    if response.status_code == 200:
        dashboards = response.json()
        for dashboard in dashboards:
            if dashboard.get('title', '').lower() == dashboard_name.lower():
                print(f"Found dashboard: {dashboard.get('title')} (UID: {dashboard.get('uid')})")
                return dashboard.get('uid')
        if dashboards:
            print(f"No exact match found, using: {dashboards[0].get('title')} (UID: {dashboards[0].get('uid')})")
            return dashboards[0].get('uid')
    print(f"Dashboard not found: {dashboard_name}")
    return None

def get_dashboard_data(grafana_url, api_key, dashboard_uid):
    """Get dashboard data."""
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    dashboard_url = f"{grafana_url}/api/dashboards/uid/{dashboard_uid}"
    response = requests.get(dashboard_url, headers=headers)
    
    if response.status_code == 200:
        return response.json().get('dashboard', {})
    print(f"Failed to get dashboard data: {response.status_code}")
    return None

def convert_grafana_time(time_str):
    """Convert Grafana time string to Unix timestamp."""
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
            print(f"Unable to parse time string: {time_str}")
            return int(time.time() - 86400)

def extract_panel_csv(grafana_url, api_key, dashboard_data, panel, output_dir):
    """Extract CSV data from a panel."""
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
            print(f"Warning: Panel '{panel_title}' (ID: {panel_id}) cannot determine datasource")
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
            print(f"Warning: Panel '{panel_title}' (ID: {panel_id}) has no valid queries")
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
            
            print(f"Fetching data for panel '{panel_title}' (ID: {panel_id}) query {i+1}/{len(queries)}...")
            response = requests.get(csv_url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success':
                    results = data['data']['result']
                    if not results:
                        print(f"Query {i+1} returned no data points")
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
                print(f"Failed to fetch data for panel '{panel_title}' (ID: {panel_id}): {response.status_code}")
        
        if all_data_frames:
            merged_df = pd.concat(all_data_frames, axis=1)
            current_time = datetime.now()
            date_folder = current_time.strftime("%Y-%m-%d-%H:%M")
            csv_folder = os.path.join(output_dir, "grafana_csv", date_folder)
            os.makedirs(csv_folder, exist_ok=True)
            filename = f"{safe_panel_title}_panel_{panel_id}.csv"
            filepath = os.path.join(csv_folder, filename)
            merged_df.to_csv(filepath)
            print(f"Saved panel '{panel_title}' (ID: {panel_id}) data to: {filepath}")
            return filepath
        print(f"No valid data returned for panel '{panel_title}' (ID: {panel_id})")
        return None
    except Exception as e:
        print(f"Error extracting panel CSV data: {str(e)}")
        return None

def extract_all_panel_csvs(grafana_url, api_key, dashboard_name, time_range, output_dir):
    """Extract CSV data for all panels in a dashboard."""
    dashboard_uid = find_dashboard_by_name(grafana_url, api_key, dashboard_name)
    if not dashboard_uid:
        return []
    
    dashboard_data = get_dashboard_data(grafana_url, api_key, dashboard_uid)
    if not dashboard_data:
        return []
    
    if time_range:
        if "," in time_range:
            from_time, to_time = time_range.split(",")
            dashboard_data['time'] = {'from': from_time.strip(), 'to': to_time.strip()}
        else:
            dashboard_data['time'] = {'from': f'now-{time_range}', 'to': 'now'}
        print(f"Using custom time range: {dashboard_data['time']['from']} to {dashboard_data['time']['to']}")
    
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

# --- Data Analysis ---
def load_data(file_path):
    """Load CSV data into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_data(df):
    """Analyze numerical columns in a DataFrame."""
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
        print(f"Error analyzing data: {e}")
        return None

def plot_trend(analysis_result, output_path, title, font_prop=None):
    """Plot trend analysis for numerical columns."""
    df = analysis_result["df"]
    stats = analysis_result["stats"]
    plt.figure(figsize=(14, 10))
    time_col = df.columns[0]
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        plt.plot(df[time_col], df[col], 'o-', markersize=3, linewidth=1, alpha=0.7, label=col)
        plt.axhline(y=stats[col]["mean"], linestyle='--', alpha=0.5, label=f'{col} Mean: {stats[col]["mean"]:.2f}')
    
    plt.title(f'{title} - Trend Analysis', fontsize=16, pad=20, fontproperties=font_prop)
    plt.xlabel('Time', fontsize=12, fontproperties=font_prop)
    plt.ylabel('Value', fontsize=12, fontproperties=font_prop)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', prop=font_prop)
    plt.xticks(rotation=45)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
    try:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Trend plot saved to: {output_path}")
    except Exception as e:
        print(f"Error saving trend plot: {e}")
    finally:
        plt.close()

def plot_histogram(analysis_result, output_path, title, font_prop=None):
    """Plot histogram for numerical columns."""
    df = analysis_result["df"]
    stats = analysis_result["stats"]
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    num_cols = len(numeric_columns)
    
    if num_cols == 0:
        return
    
    fig,axes = plt.subplots(1, num_cols, figsize=(6*num_cols, 8))
    if num_cols == 1:
        axes = [axes]
    
    for i, col in enumerate(numeric_columns):
        data = df[col].dropna().values
        if len(data) == 0:
            axes[i].text(0.5, 0.5, "No valid data", ha='center', va='center', fontproperties=font_prop)
        elif len(np.unique(data)) <= 1:
            axes[i].axvline(data[0], color='b', linewidth=2)
            axes[i].text(data[0], 0.5, f"Fixed value: {data[0]}", ha='center', va='center', fontproperties=font_prop)
        else:
            try:
                min_val, max_val = np.min(data), np.max(data)
                bins = [min_val - 0.5, min_val + 0.5] if np.isclose(min_val, max_val) else np.linspace(min_val, max_val, 31)
                axes[i].hist(data, bins=bins, alpha=0.7)
            except Exception as e:
                print(f"Error plotting histogram ({col}): {e}")
                axes[i].text(0.5, 0.5, f"Unable to plot histogram: {str(e)}", ha='center', va='center', fontproperties=font_prop)
        
        if col in stats:
            if "mean" in stats[col] and not np.isnan(stats[col]["mean"]):
                axes[i].axvline(stats[col]["mean"], color='r', linestyle='--', label=f'Mean: {stats[col]["mean"]:.2f}')
            if "median" in stats[col] and not np.isnan(stats[col]["median"]):
                axes[i].axvline(stats[col]["median"], color='g', linestyle='--', label=f'Median: {stats[col]["median"]:.2f}')
        
        axes[i].set_title(f'{col} Distribution', fontproperties=font_prop)
        axes[i].set_xlabel('Value', fontproperties=font_prop)
        axes[i].set_ylabel('Frequency', fontproperties=font_prop)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(prop=font_prop)
    
    plt.suptitle(f'{title} - Distribution Analysis', fontsize=16, y=0.98, fontproperties=font_prop)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.15, wspace=0.3)
    try:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Histogram saved to: {output_path}")
    except Exception as e:
        print(f"Error saving histogram: {e}")
    finally:
        plt.close()

def analyze_csv_file(csv_path, output_dir, font_prop=None):
    """Analyze a CSV file and generate trend and histogram plots."""
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
    """Analyze all CSV files in a directory structure."""
    if not os.path.exists(csv_dir):
        print(f"Error: CSV directory does not exist: {csv_dir}")
        return 1
    
    dir_items = os.listdir(csv_dir)
    date_dirs = [d for d in dir_items if os.path.isdir(os.path.join(csv_dir, d))]
    
    total_files = 0
    success_count = 0
    
    if date_dirs:
        print(f"Found {len(date_dirs)} date folders")
        for date_dir in date_dirs:
            date_csv_dir = os.path.join(csv_dir, date_dir)
            date_output_dir = os.path.join(output_dir, date_dir)
            if os.path.exists(date_output_dir) and os.listdir(date_output_dir):
                print(f"Skipping processed date folder: {date_dir}")
                continue
            
            csv_files = [f for f in os.listdir(date_csv_dir) if f.endswith('.csv')]
            if not csv_files:
                print(f"Warning: No CSV files found in {date_csv_dir}")
                continue
            
            print(f"Processing date folder: {date_dir} ({len(csv_files)} CSV files)")
            dir_success_count = 0
            for csv_file in csv_files:
                csv_path = os.path.join(date_csv_dir, csv_file)
                print(f"  Analyzing: {csv_file}")
                if analyze_csv_file(csv_path, date_output_dir, font_prop):
                    dir_success_count += 1
                    success_count += 1
            total_files += len(csv_files)
            print(f"  Completed analysis for {date_dir}: {dir_success_count}/{len(csv_files)} files processed")
    else:
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"Error: No CSV files found in {csv_dir}")
            return 1
        for csv_file in csv_files:
            csv_path = os.path.join(csv_dir, csv_file)
            print(f"Analyzing: {csv_file}")
            if analyze_csv_file(csv_path, output_dir, font_prop):
                success_count += 1
        total_files += len(csv_files)
    
    print(f"\nAnalysis completed: Successfully processed {success_count}/{total_files} CSV files")
    print(f"Results saved in: {output_dir}")
    return 0

# --- Report Generation ---
def load_config(config_path=None):
    """Load custom configuration or use default."""
    config = {"thresholds": DEFAULT_THRESHOLDS.copy(), "metric_keywords": METRIC_TYPE_KEYWORDS.copy()}
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            if "default" in user_config:
                config["default"].update(user_config["default"])
            if "thresholds" in user_config:
                config["thresholds"].update(user_config["thresholds"])
            if "metric_keywords" in user_config:
                for metric_type, keywords in user_config["metric_keywords"].items():
                    config["metric_keywords"][metric_type] = list(set(
                        config["metric_keywords"].get(metric_type, []) + keywords
                    ))
            print(f"Loaded custom configuration: {config_path}")
        except Exception as e:
            print(f"Error loading config: {e}, using default configuration")
    return config

def find_date_folders(base_dir):
    """Find date-formatted folders and sort by date."""
    if not os.path.exists(base_dir) or not os.path.isdir(base_dir):
        print(f"Error: Directory does not exist: {base_dir}")
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
                print(f"Warning: Folder name {item} matches date format but cannot be parsed")
    date_folders.sort(key=lambda x: x[1], reverse=True)
    return [folder[0] for folder in date_folders]

def get_comparison_folders(base_dir):
    """Get the two most recent folders for comparison."""
    date_folders = find_date_folders(base_dir)
    if len(date_folders) < 2:
        print(f"Error: Need at least two date folders for comparison, found {len(date_folders)}")
        return None
    return date_folders[0], date_folders[1]

def get_common_csv_files(folder1_path, folder2_path):
    """Get common CSV files between two folders."""
    if not os.path.exists(folder1_path) or not os.path.exists(folder2_path):
        return []
    files1 = set([f for f in os.listdir(folder1_path) if f.endswith('.csv')])
    files2 = set([f for f in os.listdir(folder2_path) if f.endswith('.csv')])
    return list(files1.intersection(files2))

def calculate_change_percentage(value1, value2):
    """Calculate percentage change between two values."""
    if value1 == 0:
        return float('inf') if value2 > 0 else 0.0
    return ((value2 - value1) / abs(value1)) * 100.0

def create_comparison_chart(metric_changes, output_path, title, font_prop=None):
    """Create a bar chart comparing old and new data."""
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
    plt.bar(x - width/2, old_values, width, label='Older Data', alpha=0.7, color='#1f77b4')
    plt.bar(x + width/2, new_values, width, label='Newer Data', alpha=0.7, color='#ff7f0e')
    
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
    
    plt.xlabel('Metric', fontsize=12, fontproperties=font_prop)
    plt.ylabel('Value', fontsize=12, fontproperties=font_prop)
    plt.title(f'{title} - Data Comparison', fontsize=14, fontproperties=font_prop)
    plt.xticks(x, metrics, rotation=45, ha='right', fontproperties=font_prop)
    plt.legend(prop=font_prop)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

def create_summary_chart(type_changes, output_path, font_prop=None):
    """Create a summary chart for performance changes."""
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
    plt.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Significant Increase Threshold (5%)')
    plt.axhline(y=-5, color='green', linestyle='--', alpha=0.5, label='Significant Decrease Threshold (-5%)')
    
    plt.xlabel('Metric Type', fontsize=12, fontproperties=font_prop)
    plt.ylabel('Average Change Percentage (%)', fontsize=12, fontproperties=font_prop)
    plt.title('System Performance Change by Metric Type', fontsize=14, fontproperties=font_prop)
    plt.xticks(x, types, fontproperties=font_prop)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(prop=font_prop)
    plt.ylim(min(plt.ylim()[0], -10), max(plt.ylim()[1], 15))
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

def convert_markdown_to_pdf(markdown_file, pdf_file, font_prop=None):
    """Convert Markdown file to PDF with proper Chinese font support."""
    try:
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        base_path = os.path.dirname(os.path.abspath(markdown_file))
        def replace_image_path(match):
            alt_text = match.group(1)
            rel_path = match.group(2)
            abs_path = os.path.normpath(os.path.join(base_path, rel_path))
            if not os.path.exists(abs_path):
                print(f"Warning: Image file does not exist: {abs_path}")
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
        print(f"PDF report generated successfully: {pdf_file}")
        return True
    except Exception as e:
        print(f"Error converting Markdown to PDF: {e}")
        return False

def generate_comparison_report(folder1, folder2, csv_dir, output_file, config=None, font_prop=None):
    """Generate performance comparison report and convert to PDF."""
    if config is None:
        config = load_config()
    
    folder1_path = os.path.join(csv_dir, folder1)
    folder2_path = os.path.join(csv_dir, folder2)
    common_csv_files = get_common_csv_files(folder1_path, folder2_path)
    
    if not common_csv_files:
        print("Error: No common CSV files found for comparison")
        return False
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"# Performance Test Report\n\nGenerated: {now}\n\n"
    report += "## Comparison Overview\n\n"
    report += f"This report compares performance data from two periods:\n\n"
    report += f"- **Newer Data**: {folder1}\n"
    report += f"- **Older Data**: {folder2}\n\n"
    report += f"Analyzed {len(common_csv_files)} common performance metrics.\n\n"
    
    report += "## Table of Contents\n\n"
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
            report += f"## {title}\n\nUnable to load data.\n\n"
            continue
        
        analysis1 = analyze_data(df1)
        analysis2 = analyze_data(df2)
        if analysis1 is None or analysis2 is None:
            report += f"## {title}\n\nUnable to analyze data.\n\n"
            continue
        
        stats1 = analysis1["stats"]
        stats2 = analysis2["stats"]
        report += f"## {title}\n\n"
        
        time_range1 = f"{df1.iloc[0, 0].strftime('%Y-%m-%d %H:%M:%S')} to {df1.iloc[-1, 0].strftime('%Y-%m-%d %H:%M:%S')}"
        time_range2 = f"{df2.iloc[0, 0].strftime('%Y-%m-%d %H:%M:%S')} to {df2.iloc[-1, 0].strftime('%Y-%m-%d %H:%M:%S')}"
        report += f"### Data Overview\n\n"
        report += f"- **Newer Data Time Range**: {time_range1}\n"
        report += f"- **Newer Data Points**: {len(df1)}\n"
        report += f"- **Older Data Time Range**: {time_range2}\n"
        report += f"- **Older Data Points**: {len(df2)}\n\n"
        
        report += "### Statistical Comparison\n\n"
        report += "|Metric|Older Data|Newer Data|Change|Change %|\n"
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
        report += "### Visualization Analysis\n\n"
        trend_img_path1 = f"data/{folder1}/{title}_trend.png"
        trend_img_path2 = f"data/{folder2}/{title}_trend.png"
        report += f"#### Trend Comparison\n\n"
        report += f"**Newer Data Trend**:\n\n"
        report += f"![{title} Newer Trend Analysis]({trend_img_path1})\n\n"
        report += f"**Older Data Trend**:\n\n"
        report += f"![{title} Older Trend Analysis]({trend_img_path2})\n\n"
        
        comparison_img_path = os.path.join(os.path.dirname(output_file), f"data/{folder1}/{title}_comparison.png")
        os.makedirs(os.path.dirname(comparison_img_path), exist_ok=True)
        create_comparison_chart(metric_changes, comparison_img_path, title, font_prop)
        report += f"#### Metric Comparison Chart\n\n"
        report += f"![{title} Data Comparison](data/{folder1}/{title}_comparison.png)\n\n"
        
        report += "### Performance Comparison Conclusion\n\n"
        conclusions = []
        metric_type = "Unknown"
        for mtype, keywords in config["metric_keywords"].items():
            if any(keyword in title.lower() for keyword in keywords):
                metric_type = {
                    "cpu": "CPU/负载",
                    "memory": "内存",
                    "disk": "磁盘/存储",
                    "network": "网络"
                }.get(mtype, mtype.capitalize())
                break
        
        significant_changes = [(col, change_data["change_pct"], "increased" if change_data["change_pct"] > 0 else "decreased")
                              for col, change_data in metric_changes.items() if abs(change_data["change_pct"]) > 10]
        
        if significant_changes:
            conclusions.append(f"**Significant Changes in {metric_type} Metrics**:")
            for col, change_pct, direction in significant_changes:
                conclusions.append(f"- {col} {direction} by {abs(change_pct):.2f}%")
            if metric_type == "CPU/负载":
                avg_change = sum(c[1] for c in significant_changes) / len(significant_changes)
                conclusions.append(f"\nOverall CPU load {'increased' if avg_change > 0 else 'decreased'}, "
                                  f"{'possibly indicating increased load or performance degradation. Investigate causes.' if avg_change > 0 else 'indicating potential performance improvement.'}")
            elif metric_type == "内存":
                avg_change = sum(c[1] for c in significant_changes) / len(significant_changes)
                conclusions.append(f"\nMemory usage {'increased, check for memory leaks or increased application usage.' if avg_change > 0 else 'decreased, indicating improved memory management.'}")
            elif metric_type == "磁盘/存储":
                avg_change = sum(c[1] for c in significant_changes) / len(significant_changes)
                conclusions.append(f"\nDisk I/O or usage {'increased, monitor storage performance or capacity.' if avg_change > 0 else 'decreased, indicating improved storage performance.'}")
            elif metric_type == "网络":
                avg_change = sum(c[1] for c in significant_changes) / len(significant_changes)
                conclusions.append(f"\nNetwork traffic or connections {'increased, monitor bandwidth and connection management.' if avg_change > 0 else 'decreased, indicating reduced network load.'}")
        else:
            conclusions.append(f"**No Significant Changes in {metric_type} Metrics**: Metrics show minimal change, indicating stable performance.")
        
        report += "\n".join(conclusions) + "\n\n"
        all_metrics_changes[title] = {"type": metric_type, "changes": metric_changes}
    
    report += "## Overall Conclusion\n\n"
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
    report += "### Overall Performance Change Chart\n\n"
    report += f"![Overall Performance Comparison](data/{folder1}/summary_comparison.png)\n\n"
    
    conclusions = ["Based on the comparison, system performance changes are summarized as follows:\n"]
    for metric_type, changes in type_changes.items():
        if not changes:
            continue
        avg_type_change = sum(c[1] for c in changes) / len(changes)
        if metric_type == "CPU/负载":
            conclusions.append(f"1. **CPU/Load**: {'Increased' if avg_type_change > 5 else 'Decreased' if avg_type_change < -5 else 'Stable'} "
                              f"by {abs(avg_type_change):.2f}%, {'investigate causes.' if avg_type_change > 5 else 'indicating improvement.' if avg_type_change < -5 else 'system load stable.'}")
        elif metric_type == "内存":
            conclusions.append(f"2. **Memory**: {'Increased' if avg_type_change > 5 else 'Decreased' if avg_type_change < -5 else 'Stable'} "
                              f"by {abs(avg_type_change):.2f}%, {'check memory management.' if avg_type_change > 5 else 'improved management.' if avg_type_change < -5 else 'memory usage stable.'}")
        elif metric_type == "磁盘/存储":
            conclusions.append(f"3. **Disk/Storage**: {'Increased' if avg_type_change > 5 else 'Decreased' if avg_type_change < -5 else 'Stable'} "
                              f"by {abs(avg_type_change):.2f}%, {'monitor performance.' if avg_type_change > 5 else 'improved performance.' if avg_type_change < -5 else 'storage performance stable.'}")
        elif metric_type == "网络":
            conclusions.append(f"4. **Network**: {'Increased' if avg_type_change > 5 else 'Decreased' if avg_type_change < -5 else 'Stable'} "
                              f"by {abs(avg_type_change):.2f}%, {'monitor bandwidth.' if avg_type_change > 5 else 'reduced load.' if avg_type_change < -5 else 'network performance stable.'}")
        else:
            conclusions.append(f"- **{metric_type}**: {'Increased' if avg_type_change > 5 else 'Decreased' if avg_type_change < -5 else 'Stable'} "
                              f"by {abs(avg_type_change):.2f}%.")
    
    significant_changes_count = sum(1 for changes in type_changes.values() for _, change in changes if abs(change) > 10)
    conclusions.append(f"\n**Recommendation**: {'Found' if significant_changes_count > 0 else 'No'} significant changes "
                      f"({'investigate causes and optimize.' if significant_changes_count > 0 else 'maintain current configuration.'})")
    
    report += "\n".join(conclusions) + "\n"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    pdf_file = output_file.replace('.md', '.pdf')
    convert_markdown_to_pdf(output_file, pdf_file, font_prop)
    return True

# --- Main Function ---
def main():
    """Main function to execute the combined workflow."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_default = os.path.join(script_dir, "analysis_config.json")
    
    if os.path.exists(config_file_default):
        config = load_config(config_file_default)
        default_config = config.get("basic", {})
    else:
        default_config = DEFAULT_BASIC.copy()
    parser = argparse.ArgumentParser(description="Combined script for Grafana data extraction, analysis, and report generation.")
    parser.add_argument("--mode", choices=['all', 'extract', 'analyze', 'report'], default='all',
                        help="Execution mode: all (default), extract, analyze, or report")
    parser.add_argument("--grafana-url", default="http://192.168.8.244:3000", help="Grafana URL")
    parser.add_argument("--username", default="admin", help="Grafana username")
    parser.add_argument("--password", default="password", help="Grafana password (use with caution)")
    parser.add_argument("--dashboard", default="VM-exporter", help="Grafana dashboard name")
    parser.add_argument("--time-range", help="Time range (e.g., '1h' or '2023-01-01T00:00:00Z,2023-01-02T00:00:00Z')")
    parser.add_argument("--csv-dir", help="CSV files directory", default=None)
    parser.add_argument("--output-dir", help="Output directory for analysis", default=None)
    parser.add_argument("--output-file", help="Output Markdown file for report", default=None)
    parser.add_argument("--config", help="Custom configuration file", default=None)
    parser.add_argument("--folder1", help="Newer data folder (optional, default: latest)", default=None)
    parser.add_argument("--folder2", help="Older data folder (optional, default: second latest)", default=None)
    args = parser.parse_args()
    
    if args.config:
        config = load_config(args.config)
        default_config = config.get("basic", {})
    
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
        print("Error: Font setup failed, exiting")
        return 1
    
    print(f"use default grafana_url={grafana_url}, username={username}, dashboard={dashboard_name}")

    if args.mode in ['all', 'extract']:
        use_default = input(f"Use default credentials ({args.username}@{args.grafana_url})? (Y/n): ").lower() != "n"
        grafana_url = args.grafana_url
        username = args.username
        password = args.password
        dashboard_name = args.dashboard
        time_range = args.time_range
        
        if not use_default:
            grafana_url = input(f"Enter Grafana URL (default: {grafana_url}): ") or grafana_url
            username = input(f"Enter Grafana username (default: {username}): ") or username
            password = getpass.getpass("Enter Grafana password: ")
        
        if not time_range:
            use_custom_time = input("Use custom time range? (y/N): ").lower() == "y"
            if use_custom_time:
                time_range = input("Enter time range (default: 1h): ") or "1h"
        
        api_key = get_grafana_api_key(grafana_url, username, password, script_dir)
        if not api_key:
            print("Failed to get Grafana API key, exiting")
            return 1
        
        start_time = time.time()
        csv_files = extract_all_panel_csvs(grafana_url, api_key, dashboard_name, time_range, script_dir)
        end_time = time.time()
        
        if csv_files:
            print("\nSuccessfully exported CSV files:")
            for file in csv_files:
                print(f"- {file}")
            print(f"\nExported {len(csv_files)} panels in {end_time - start_time:.2f} seconds")
        else:
            print("Failed to export any CSV files")
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
                print("Error: Insufficient date folders for comparison")
                return 1
            folder1, folder2 = folders
        
        folder1_path = os.path.join(csv_dir, folder1)
        folder2_path = os.path.join(csv_dir, folder2)
        if not os.path.exists(folder1_path) or not os.path.exists(folder2_path):
            print(f"Error: Specified folders do not exist: {folder1_path} or {folder2_path}")
            return 1
        
        print(f"Generating performance comparison report...")
        print(f"Using config: {args.config}")
        print(f"Newer data folder: {folder1}")
        print(f"Older data folder: {folder2}")
        print(f"Output report file: {output_file}")
        
        if generate_comparison_report(folder1, folder2, csv_dir, output_file, config, font_prop):
            print(f"Comparison report generated successfully: {output_file}")
        else:
            print("Failed to generate comparison report")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

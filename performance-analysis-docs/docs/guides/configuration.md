# Configuration Guide for Performance Analysis Project

This document explains how to configure the project settings, including environment variables and configuration files.

## Configuration Files

The project uses a configuration file to manage various settings. The default configuration file is named `config.json` and should be placed in the root directory of the project. Below is an example of the configuration file structure:

```json
{
    "grafana_url": "http://your-grafana-url",
    "username": "your-username",
    "password": "your-password",
    "dashboard": "your-dashboard-name",
    "time_range": "1h",
    "csv_dir": "path/to/csv/files",
    "output_dir": "path/to/output/directory"
}
```

### Parameters

- **grafana_url**: The URL of your Grafana instance.
- **username**: Your Grafana username.
- **password**: Your Grafana password (ensure to handle this securely).
- **dashboard**: The name of the Grafana dashboard you want to analyze.
- **time_range**: The time range for the data you want to extract (e.g., `1h`, `24h`, etc.).
- **csv_dir**: The directory where CSV files are stored.
- **output_dir**: The directory where output files will be saved.

## Environment Variables

In addition to the configuration file, you can also set environment variables to override the default settings. The following environment variables are supported:

- `GRAFANA_URL`: Overrides the `grafana_url` in the configuration file.
- `GRAFANA_USERNAME`: Overrides the `username` in the configuration file.
- `GRAFANA_PASSWORD`: Overrides the `password` in the configuration file.
- `DASHBOARD_NAME`: Overrides the `dashboard` in the configuration file.
- `TIME_RANGE`: Overrides the `time_range` in the configuration file.
- `CSV_DIR`: Overrides the `csv_dir` in the configuration file.
- `OUTPUT_DIR`: Overrides the `output_dir` in the configuration file.

## Example Usage

To use the configuration in your scripts, you can load the configuration file as follows:

```python
import json

with open('config.json') as config_file:
    config = json.load(config_file)

grafana_url = config['grafana_url']
username = config['username']
password = config['password']
```

Alternatively, you can access the environment variables using the `os` module:

```python
import os

grafana_url = os.getenv('GRAFANA_URL', config['grafana_url'])
username = os.getenv('GRAFANA_USERNAME', config['username'])
password = os.getenv('GRAFANA_PASSWORD', config['password'])
```

## Conclusion

Proper configuration of the project settings is essential for successful data extraction and analysis. Ensure that all parameters are set correctly in the configuration file or through environment variables before running the project.
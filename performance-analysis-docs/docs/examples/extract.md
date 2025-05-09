# Extracting Data from Various Sources

This document illustrates how to extract data from various sources using the functions provided in the project. Below are practical examples and best practices for data extraction.

## Example 1: Extracting Data from a CSV File

To extract data from a CSV file, you can use the `load_data` function. Here’s a simple example:

```python
import pandas as pd

def extract_data_from_csv(file_path):
    """Extract data from a CSV file."""
    df = pd.read_csv(file_path)
    return df

# Usage
csv_file_path = 'path/to/your/data.csv'
data = extract_data_from_csv(csv_file_path)
print(data.head())
```

## Example 2: Extracting Data from a Database

If you need to extract data from a database, you can use the following approach:

```python
import sqlite3

def extract_data_from_db(db_path, query):
    """Extract data from a SQLite database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Usage
database_path = 'path/to/your/database.db'
sql_query = 'SELECT * FROM your_table'
data = extract_data_from_db(database_path, sql_query)
print(data.head())
```

## Example 3: Extracting Data from an API

To extract data from an API, you can use the `requests` library. Here’s an example:

```python
import requests

def extract_data_from_api(api_url):
    """Extract data from a REST API."""
    response = requests.get(api_url)
    data = response.json()
    return data

# Usage
api_endpoint = 'https://api.example.com/data'
data = extract_data_from_api(api_endpoint)
print(data)
```

## Best Practices

- Always validate the data after extraction to ensure it meets your expectations.
- Handle exceptions and errors gracefully to avoid crashes during data extraction.
- Consider using logging to track the extraction process and any issues that arise.

By following these examples and best practices, you can effectively extract data from various sources for analysis and reporting.
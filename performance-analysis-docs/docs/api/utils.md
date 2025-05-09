# Utility Functions Documentation

This document describes the utility functions that assist in data processing and manipulation within the performance analysis project. These functions are designed to simplify common tasks and enhance the overall functionality of the analysis and reporting processes.

## Utility Functions

### 1. `load_data(file_path: str) -> DataFrame`
Loads CSV data from the specified file path into a pandas DataFrame.

**Parameters:**
- `file_path`: The path to the CSV file to be loaded.

**Returns:**
- A pandas DataFrame containing the loaded data.

**Example:**
```python
df = load_data('data/sample_data.csv')
```

### 2. `save_data(df: DataFrame, file_path: str) -> None`
Saves the provided DataFrame to a specified CSV file.

**Parameters:**
- `df`: The pandas DataFrame to be saved.
- `file_path`: The path where the CSV file will be saved.

**Example:**
```python
save_data(df, 'data/output_data.csv')
```

### 3. `calculate_statistics(df: DataFrame) -> dict`
Calculates basic statistics (mean, median, standard deviation) for numerical columns in the DataFrame.

**Parameters:**
- `df`: The pandas DataFrame for which statistics will be calculated.

**Returns:**
- A dictionary containing the calculated statistics.

**Example:**
```python
stats = calculate_statistics(df)
```

### 4. `filter_data(df: DataFrame, condition: str) -> DataFrame`
Filters the DataFrame based on a specified condition.

**Parameters:**
- `df`: The pandas DataFrame to be filtered.
- `condition`: A string representing the condition for filtering.

**Returns:**
- A new DataFrame containing only the rows that meet the condition.

**Example:**
```python
filtered_df = filter_data(df, 'column_name > 10')
```

### 5. `merge_dataframes(df1: DataFrame, df2: DataFrame, on: str) -> DataFrame`
Merges two DataFrames on a specified column.

**Parameters:**
- `df1`: The first DataFrame to merge.
- `df2`: The second DataFrame to merge.
- `on`: The column name on which to merge the DataFrames.

**Returns:**
- A new DataFrame resulting from the merge.

**Example:**
```python
merged_df = merge_dataframes(df1, df2, 'id')
```

## Conclusion

These utility functions are essential for efficient data handling and processing in the performance analysis project. By utilizing these functions, users can streamline their workflows and focus on analyzing and reporting their data effectively.
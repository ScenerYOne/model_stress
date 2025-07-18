from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import sys

# Set console encoding to utf-8
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

# Define paths
COMBINED_PATH = Path('Combined/eda_ppg')
OUTPUT_PATH = Path('Output/Before_label')

# Create target directory if it doesn't exist
COMBINED_PATH.mkdir(parents=True, exist_ok=True)

def load_all_csv_from_dir(directory: Path) -> pd.DataFrame:
    """Load all CSV files from directory and combine them into a single DataFrame"""
    print(f"Loading CSV files from {directory}")
    
    # Create empty DataFrame
    df = pd.DataFrame()
    
    # Get list of CSV files
    csv_files = list(directory.glob('*.csv'))
    print(f"Found {len(csv_files)} CSV files")
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"Processing {csv_file.name}")
        try:
            temp_df = pd.read_csv(csv_file)
            print(f"  - Loaded {len(temp_df)} rows")
            
            # Check for empty values
            empty_count = temp_df.isna().sum().sum()
            print(f"  - Found {empty_count} empty values")
            
            df = pd.concat([df, temp_df], ignore_index=True)
        except Exception as e:
            print(f"  - Error loading {csv_file}: {e}")
    
    if not df.empty:
        # Display summary of empty values by column before cleaning
        print("\nEmpty values by column before cleaning:")
        missing_by_column = df.isnull().sum()
        for col, count in missing_by_column.items():
            if count > 0:
                print(f"  - {col}: {count} empty values ({count/len(df)*100:.2f}%)")
        
        # Convert DateTime column to datetime format
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        
        print(f"DataFrame after DateTime conversion has {len(df)} rows and {len(df.columns)} columns")
    else:
        print("No data loaded from CSV files")
    
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in a smarter way"""
    if df.empty:
        print("DataFrame is empty, nothing to clean")
        return df
    
    print(f"Original DataFrame has {len(df)} rows and {len(df.columns)} columns")
    
    # Identify columns with 100% missing values
    fully_missing_cols = []
    for col in df.columns:
        if df[col].isna().sum() == len(df):
            fully_missing_cols.append(col)
    
    print(f"Found {len(fully_missing_cols)} columns with 100% missing values: {fully_missing_cols}")
    
    # Set strategy: Drop columns that are 100% missing
    df_clean = df.drop(columns=fully_missing_cols)
    print(f"Dropped {len(fully_missing_cols)} completely empty columns")
    
    # Identify key data columns (DateTime and id should always be present)
    key_cols = ['DateTime', 'id']
    
    # Check for missing values in key columns
    key_missing = df_clean[key_cols].isna().any(axis=1)
    if key_missing.any():
        rows_to_drop = key_missing.sum()
        df_clean = df_clean[~key_missing]
        print(f"Dropped {rows_to_drop} rows with missing values in key columns (DateTime, id)")
    
    # Find rows with less than 50% missing values
    missing_percentage = df_clean.isna().mean(axis=1) * 100
    print("\nDistribution of missing values per row:")
    print(f"  - Min: {missing_percentage.min():.2f}%")
    print(f"  - 25th percentile: {missing_percentage.quantile(0.25):.2f}%")
    print(f"  - Median: {missing_percentage.median():.2f}%")
    print(f"  - 75th percentile: {missing_percentage.quantile(0.75):.2f}%")
    print(f"  - Max: {missing_percentage.max():.2f}%")
    
    # Keep rows with less than X% missing values
    threshold = 50  # Adjust this threshold as needed
    keep_rows = missing_percentage < threshold
    rows_to_keep = keep_rows.sum()
    
    if rows_to_keep == 0:
        print(f"No rows have less than {threshold}% missing values. Adjusting threshold.")
        # Find a threshold that will keep at least some data
        for test_threshold in [60, 70, 80, 90, 95]:
            keep_rows = missing_percentage < test_threshold
            rows_to_keep = keep_rows.sum()
            if rows_to_keep > 0:
                threshold = test_threshold
                print(f"Adjusted threshold to {threshold}% missing values, which keeps {rows_to_keep} rows")
                break
    
    df_clean = df_clean[keep_rows]
    print(f"Kept {rows_to_keep} rows with less than {threshold}% missing values")
    
    # Remove duplicate rows based on DateTime and id
    original_len = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['DateTime', 'id'])
    rows_removed = original_len - len(df_clean)
    print(f"Removed {rows_removed} duplicate rows")
    
    # Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    print(f"Final DataFrame has {len(df_clean)} rows and {len(df_clean.columns)} columns")
    
    return df_clean

def examine_data(df: pd.DataFrame) -> None:
    """Examine the data to understand patterns of missing values"""
    if df.empty:
        print("DataFrame is empty, nothing to examine")
        return
    
    print("\n=== Data Examination ===")
    
    # Check for patterns in missing values - do they cluster by ID?
    if 'id' in df.columns:
        ids = df['id'].unique()
        print(f"Found {len(ids)} unique IDs")
        
        for id_val in ids[:min(5, len(ids))]:  # Show stats for first few IDs
            id_data = df[df['id'] == id_val]
            missing_pct = id_data.isna().mean().mean() * 100
            print(f"ID {id_val}: {len(id_data)} rows, {missing_pct:.2f}% missing values")
    
    # Check for patterns in missing values over time
    if 'DateTime' in df.columns:
        df_sorted = df.sort_values('DateTime')
        # Create time bins (e.g., hourly)
        df_sorted['TimeGroup'] = df_sorted['DateTime'].dt.floor('H')
        
        time_groups = df_sorted['TimeGroup'].unique()
        print(f"Found {len(time_groups)} unique time groups")
        
        if len(time_groups) > 0:
            print("Missing values by time (first 5 time groups):")
            for time_group in time_groups[:min(5, len(time_groups))]:
                time_data = df_sorted[df_sorted['TimeGroup'] == time_group]
                missing_pct = time_data.isna().mean().mean() * 100
                print(f"Time {time_group}: {len(time_data)} rows, {missing_pct:.2f}% missing values")
            
            # Clean up temporary column
            df_sorted = df_sorted.drop(columns=['TimeGroup'])
    
    # Count how often columns are missing together
    print("\nExamining patterns of missing values across columns...")
    na_counts = df.isna().sum()
    columns_with_na = [col for col, count in na_counts.items() if count > 0]
    
    if len(columns_with_na) > 1:
        print(f"Found {len(columns_with_na)} columns with missing values")
        
        # Check if there are patterns in how columns are missing together
        na_patterns = {}
        for i, col1 in enumerate(columns_with_na[:min(5, len(columns_with_na))]):
            for col2 in columns_with_na[i+1:min(i+6, len(columns_with_na))]:
                both_na = (df[col1].isna() & df[col2].isna()).sum()
                either_na = (df[col1].isna() | df[col2].isna()).sum()
                if either_na > 0:
                    overlap_pct = (both_na / either_na) * 100
                    print(f"{col1} and {col2}: {overlap_pct:.2f}% overlap in missing values")

def main():
    # Load all data from directory
    print("\n=== Loading data from CSV files ===")
    all_data = load_all_csv_from_dir(COMBINED_PATH)
    
    if all_data.empty:
        print("No data found, exiting")
        return
    
    # Examine data to understand patterns of missing values
    examine_data(all_data)
    
    # Handle missing values in a more nuanced way
    print("\n=== Handling missing values ===")
    cleaned_data = handle_missing_values(all_data)
    
    if cleaned_data.empty:
        print("No data left after cleaning, exiting")
        return
    
    # Add timestamp to filename to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_PATH  / f'Data_ea_pg.csv'
    
    try:
        # Save cleaned data with all original columns intact
        cleaned_data.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        print(f"Final dataset has {len(cleaned_data)} rows and {len(cleaned_data.columns)} columns")
    except IOError as e:
        print(f"Error writing file: {e}")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return

if __name__ == '__main__':
    main()
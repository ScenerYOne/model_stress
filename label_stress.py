import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta

# Define paths - FIX: corrected the file path
PROCESSED_PATH = Path('Output/Before_label')
CSV_FILE = PROCESSED_PATH / 'Data_ea_pg.csv'
LABEL_FILE = Path('label_stress.json')
OUTPUT_PATH = Path('Output/After_label')

# Create output directory if it doesn't exist
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

def load_json(file_path: Path) -> list:
    """Load stress labels from JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    print(f"JSON file structure type: {type(data)}")
    if isinstance(data, dict) and 'label_stress' in data and isinstance(data['label_stress'], list):
        return data['label_stress']
    elif isinstance(data, list):
        return data
    else:
        print(f"JSON structure keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        raise ValueError("JSON structure unexpected. Expecting dict with 'label_stress' key containing a list.")


def main():
    # Check if file exists before loading
    if not CSV_FILE.exists():
        print(f"Error: File not found at {CSV_FILE}")
        print("Checking for available files in directory...")
        available_files = list(PROCESSED_PATH.glob('*.csv'))
        if available_files:
            print(f"Available CSV files: {[f.name for f in available_files]}")
            # Try to use the first available file
            corrected_file = available_files[0]
            print(f"Using {corrected_file} instead")
            df = pd.read_csv(corrected_file)
        else:
            print("No CSV files found in the directory.")
            return
    else:
        # Load the data
        print(f"Loading data from {CSV_FILE}")
        df = pd.read_csv(CSV_FILE)
    
    initial_count = len(df)
    print(f"Initial row count: {initial_count}")
    
    # Convert DateTime to datetime format
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Check for missing values before dropping
    na_counts = df.isna().sum()
    print(f"Missing value counts before dropna:\n{na_counts}")
    
    # Remove rows with missing values - CAREFUL: THIS COULD REMOVE ALL ROWS
    # Modify to only drop rows with NaN in critical columns
    critical_columns = ['DateTime', 'id']  # Define columns that are essential
    df = df.dropna(subset=critical_columns)
    print(f"Row count after dropna: {len(df)}")
    
    # Load stress labels
    stress_labels = load_json(LABEL_FILE)
    print(f"Loaded {len(stress_labels)} stress labels")
    
    # Create a dictionary for quick lookup by id
    # Add debugging to see ID formats
    print(f"First few labels: {stress_labels[:3] if len(stress_labels) > 0 else 'Empty'}")
    stress_dict = {item['id']: item for item in stress_labels}
    print(f"Stress dict keys (first 5): {list(stress_dict.keys())[:5]}")
    
    # Print CSV IDs for comparison 
    csv_ids = df['id'].unique()
    print(f"CSV unique IDs (first 5): {list(csv_ids)[:5]}")
    print(f"Total unique IDs in CSV: {len(csv_ids)}")
    
    # Check if there are any matching IDs
    matching_ids = [id for id in csv_ids if id in stress_dict]
    print(f"Matching IDs count: {len(matching_ids)}")
    if len(matching_ids) > 0:
        print(f"Example matching IDs: {matching_ids[:5]}")
    else:
        print("No matching IDs found between CSV and JSON!")
        
        # Try alternative case-insensitive matching if strict matching fails
        csv_ids_lower = [str(id).lower() for id in csv_ids]
        json_ids_lower = [str(id).lower() for id in stress_dict.keys()]
        
        common_ids = set(csv_ids_lower).intersection(set(json_ids_lower))
        if common_ids:
            print(f"Found {len(common_ids)} matches when comparing case-insensitive IDs")
    
    # Add stress column with default NaN values
    df['stress'] = pd.NA
    
    # Process each unique id in the dataframe
    labeled_count_by_subject = {}
    for subject_id in df['id'].unique():
        if subject_id not in stress_dict:
            print(f"Warning: No stress label found for subject {subject_id}")
            continue
            
        # Get data for this subject
        subject_data = df[df['id'] == subject_id]
        
        if len(subject_data) == 0:
            print(f"No data found for subject {subject_id}")
            continue
            
        # Find start time for this subject
        start_time = subject_data['DateTime'].min()
        print(f"Subject {subject_id} start time: {start_time}")
        
        # Define time windows for labeling - FIX: added missing comma
        try:
            windows = [
                (start_time, start_time + timedelta(minutes=5), "normal"),
                (start_time + timedelta(minutes=5), start_time + timedelta(minutes=10), "normal"),
                (start_time + timedelta(minutes=10), start_time + timedelta(minutes=13), stress_dict[subject_id]['section1']),
                (start_time + timedelta(minutes=13), start_time + timedelta(minutes=15), "normal"),
                (start_time + timedelta(minutes=15), start_time + timedelta(minutes=18), stress_dict[subject_id]['section2']),
                (start_time + timedelta(minutes=18), start_time + timedelta(minutes=20), "normal"),
                (start_time + timedelta(minutes=20), start_time + timedelta(minutes=25), stress_dict[subject_id]['section3']),  # Added missing comma
                (start_time + timedelta(minutes=25), start_time + timedelta(minutes=30), "normal")
            ]
        except KeyError as e:
            print(f"Error with subject {subject_id}: Missing key {e} in label data")
            print(f"Available keys for this subject: {list(stress_dict[subject_id].keys())}")
            continue
        
        # Apply labels for each time window
        subject_labeled_count = 0
        for start, end, label in windows:
            mask = (df['id'] == subject_id) & (df['DateTime'] >= start) & (df['DateTime'] < end)
            rows_affected = mask.sum()
            df.loc[mask, 'stress'] = label
            subject_labeled_count += rows_affected
            print(f"Subject {subject_id}, window {start} to {end}: {rows_affected} rows labeled as '{label}'")
        
        labeled_count_by_subject[subject_id] = subject_labeled_count
    
    # Save the labeled data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_PATH / f"labeled_stress_data.csv"
    df.to_csv(output_file, index=False)
    print(f"Labeled data saved to {output_file}")
    
    # Print some statistics
    labeled_count = df['stress'].notna().sum()
    total_count = len(df)
    if total_count > 0:
        print(f"Labeled {labeled_count} out of {total_count} records ({labeled_count/total_count*100:.2f}%)")
        
        # Distribution of stress levels
        print("\nStress level distribution:")
        stress_dist = df['stress'].value_counts(dropna=False)
        for level, count in stress_dist.items():
            print(f"{level}: {count} ({count/total_count*100:.2f}%)")
            
        # Per-subject summary
        print("\nLabeled rows by subject:")
        for subject_id, count in labeled_count_by_subject.items():
            print(f"Subject {subject_id}: {count} rows labeled")
    else:
        print("WARNING: Final DataFrame is empty - no records were processed!")

if __name__ == "__main__":
    main()
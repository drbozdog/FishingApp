import csv
import os
import asyncio
import hashlib
import json
from datetime import datetime
from typing import Dict, Any, Set
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add parent directory to path

# Assuming geocoding_agent.py is in the same directory or accessible via PYTHONPATH
from geocoding_responses import run as geocode_with_responses, get_river_segment_coordinates # Import run from geocoding_responses and get_river_segment_coordinates
from agent import OpenAIManager # Import OpenAIManager for token tallying

# --- Configuration ---
INPUT_CSV_FILE = "data/ANPA_habitats_contractate_2025_full.csv"
OUTPUT_CSV_FILE = "data/geocoded_output.csv"
PROCESSING_LOG_FILE = "data/processing_status.csv" # Logs attempts and status
MAX_ROWS = 30  # Maximum number of rows to process

# Columns to create a unique identifier for each row if no single ID column exists
# Update these to match the actual column names in your CSV
ROW_IDENTIFIER_COLUMNS = ['County', 'Habitat', 'Limits']

# --- Helper Functions ---

def get_row_identifier(row: Dict[str, str]) -> str:
    """Creates a unique identifier for a row."""
    identifier_parts = [str(row.get(col, '')) for col in ROW_IDENTIFIER_COLUMNS]
    return hashlib.md5("-".join(identifier_parts).encode('utf-8')).hexdigest()

def load_processed_rows(log_file_path: str) -> Set[str]:
    """Loads identifiers of rows that have already been processed (succeeded or failed)."""
    processed_ids = set()
    if not os.path.exists(log_file_path):
        return processed_ids
    try:
        log_df = pd.read_csv(log_file_path)
        processed_ids = set(log_df[log_df['status'].isin(['succeeded', 'failed'])]['row_identifier'])
    except Exception as e:
        print(f"Error loading processing log '{log_file_path}': {e}")
    return processed_ids

def log_processing_status(log_file_path: str, row_identifier: str, status: str, 
                          message: str, original_row_data: Dict[str, str], 
                          geocoded_output: str = ""):
    """Logs the processing status of a row to the status log CSV."""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'row_identifier': row_identifier,
        'status': status,
        'message': message,
        'geocoded_output': geocoded_output
    }
    log_entry.update(original_row_data)
    
    log_df = pd.DataFrame([log_entry])
    mode = 'a' if os.path.exists(log_file_path) else 'w'
    header = not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0
    log_df.to_csv(log_file_path, mode=mode, header=header, index=False)

# --- Main ETL Logic ---
def main_etl_process():
    print(f"Starting ETL process...")
    print(f"Input CSV: {INPUT_CSV_FILE}")
    print(f"Output CSV: {OUTPUT_CSV_FILE}")
    print(f"Processing Log: {PROCESSING_LOG_FILE}")

    processed_row_ids = load_processed_rows(PROCESSING_LOG_FILE)
    print(f"Loaded {len(processed_row_ids)} already processed row identifiers.")

    try:
        # Read input CSV with pandas
        df = pd.read_csv(INPUT_CSV_FILE)
        
        # Initialize output DataFrame if it exists
        if os.path.exists(OUTPUT_CSV_FILE):
            output_df = pd.read_csv(OUTPUT_CSV_FILE)
        else:
            output_df = pd.DataFrame()

        rows_processed_this_session = 0
        rows_skipped = 0
        rows_succeeded = 0
        rows_failed = 0
        all_geocoding_logs = []

        for index, row in df.iterrows():
            if rows_processed_this_session >= MAX_ROWS:
                print(f"\nReached maximum row limit of {MAX_ROWS}. Stopping processing.")
                break

            row_data = row.to_dict()
            row_identifier = get_row_identifier(row_data)

            if row_identifier in processed_row_ids:
                rows_skipped += 1
                continue

            print(f"Processing row {index + 1} (ID: {row_identifier})...")
            rows_processed_this_session += 1

            try:
                # Extract coordinates using the river segment coordinates function
                coordinates = get_river_segment_coordinates(
                    county=row_data.get('County', 'N/A'),
                    river_name=row_data.get('Habitat', 'N/A'),
                    segment=row_data.get('Limits', 'N/A'),
                    length=row_data.get('Length_surface', 'N/A'),
                    country='Romania'
                )

                # Prepare output row
                output_row = row_data.copy()
                output_row.update({
                    'start_point_latitude': coordinates['start_point_latitude'],
                    'start_point_longitude': coordinates['start_point_longitude'],
                    'end_point_latitude': coordinates['end_point_latitude'],
                    'end_point_longitude': coordinates['end_point_longitude'],
                    'raw_geocoding_result': json.dumps(coordinates),
                    'processing_status': 'succeeded',
                    'error_message': ''
                })

                rows_succeeded += 1
                print(f"  Row {index + 1} (ID: {row_identifier}) processed successfully.")

            except Exception as e:
                print(f"  Error processing row {index + 1} (ID: {row_identifier}): {e}")
                output_row = row_data.copy()
                output_row.update({
                    'start_point_latitude': None,
                    'start_point_longitude': None,
                    'end_point_latitude': None,
                    'end_point_longitude': None,
                    'raw_geocoding_result': '',
                    'processing_status': 'failed',
                    'error_message': str(e)
                })
                rows_failed += 1

            # Append to output DataFrame
            output_df = pd.concat([output_df, pd.DataFrame([output_row])], ignore_index=True)
            
            # Save after each row to ensure we don't lose progress
            output_df.to_csv(OUTPUT_CSV_FILE, index=False)

            # Log status
            log_processing_status(
                PROCESSING_LOG_FILE,
                row_identifier,
                output_row['processing_status'],
                output_row['error_message'],
                row_data,
                output_row['raw_geocoding_result']
            )
            processed_row_ids.add(row_identifier)

    except FileNotFoundError:
        print(f"Error: Input CSV file not found: {INPUT_CSV_FILE}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during ETL processing: {e}")
        import traceback
        traceback.print_exc()

    print("\nETL process completed.")
    print(f"Summary:")
    print(f"  Rows processed this session: {rows_processed_this_session}")
    print(f"  Rows skipped (already processed): {rows_skipped}")
    print(f"  Rows succeeded: {rows_succeeded}")
    print(f"  Rows failed: {rows_failed}")

    if all_geocoding_logs:
        print("\nCalculating token usage for this session...")
        manager = OpenAIManager()
        manager.tally(all_geocoding_logs)
    else:
        print("\nNo new API calls made this session, so no token usage to display.")

if __name__ == "__main__":
    main_etl_process() 
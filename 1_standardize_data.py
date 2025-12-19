import pandas as pd
import glob
import os

# 1. Setup paths
DATA_FOLDER = './data'
MASTER_FILE = 'master_data.csv'

def standardize_work():
    # Find all CSV files in the data folder
    all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    
    if not all_files:
        print("Error: No CSV files found in /data folder!")
        return

    li = []

    # 2. Load and Basic Clean
    for filename in all_files:
        # Use unicode_escape because some names have special characters
        df = pd.read_csv(filename, index_col=None, header=0, encoding='unicode_escape')
        
        # We only need these specific professional columns for now
        # HST/AST = Shots on Target (Most important stat for bots)
        cols_we_need = [
            'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 
            'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'B365H', 'B365D', 'B365A'
        ]
        
        # Check if the CSV actually has these columns (some very old ones might miss HST)
        available_cols = [c for c in cols_we_need if c in df.columns]
        df = df[available_cols]
        
        li.append(df)

    # 3. Merge into one Master Frame
    master_df = pd.concat(li, axis=0, ignore_index=True)

    # 4. Standardize Dates (Crucial for Ubuntu/Python sorting)
    # This handles both 01/12/25 and 2025-12-01 formats
    master_df['Date'] = pd.to_datetime(master_df['Date'], dayfirst=True, errors='coerce')
    
    # Remove any rows where the date failed or match was cancelled
    master_df = master_df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])

    # 5. Team Name Normalization
    # This ensures "Oxford Utd" and "Oxford United" are treated as the same team
    def normalize_names(name):
        return name.replace('Utd', 'United').replace('City', 'City').strip()

    master_df['HomeTeam'] = master_df['HomeTeam'].apply(normalize_names)
    master_df['AwayTeam'] = master_df['AwayTeam'].apply(normalize_names)

    # 6. Sort by Date
    master_df = master_df.sort_values(by='Date')

    # 7. Save the Clean Dataset
    master_df.to_csv(MASTER_FILE, index=False)
    
    print(f"--- Task 2 Complete ---")
    print(f"Total Matches Standardized: {len(master_df)}")
    print(f"Unique Teams Found: {master_df['HomeTeam'].nunique()}")
    print(f"Master file saved as: {MASTER_FILE}")

if __name__ == "__main__":
    standardize_work()
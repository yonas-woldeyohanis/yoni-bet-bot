import pandas as pd
import requests
import os

# The URL for the current season's League One data
URL = "https://www.football-data.co.uk/mmz4281/2526/E2.csv"
MASTER_PATH = "master_data.csv"

def run_auto_update():
    print("🔄 Fetching latest weekly results...")
    try:
        # 1. Download the latest season data
        new_data = pd.read_csv(URL)
        
        # 2. Basic cleaning
        new_data['Date'] = pd.to_datetime(new_data['Date'], dayfirst=True)
        
        # 3. Save as current season file
        new_data.to_csv("./data/Season2025-2026.csv", index=False)
        print("✅ Season 2025-2026 data updated.")
        
        # 4. Trigger the retraining chain
        print("🧠 Retraining AI with new match data...")
        os.system("python3 1_standardize_data.py")
        os.system("python3 2_elo_engine.py")
        os.system("python3 8_pro_preprocessor.py")
        os.system("python3 9_train_pro_models.py")
        os.system("python3 11_create_snapshot.py")
        print("🚀 ALL SYSTEMS UPDATED AND AI RETRAINED.")
        
    except Exception as e:
        print(f"❌ Update failed: {e}")

if __name__ == "__main__":
    run_auto_update()
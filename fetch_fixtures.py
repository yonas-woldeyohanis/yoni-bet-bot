import pandas as pd
import json

def generate_fixtures():
    print("search 🕵️ Fetching fixtures from public data...")
    # We use a public sports data source that provides the season schedule
    url = "https://fixturedownload.com/download/epl-2025-E2.csv" # Standard public schedule link
    
    try:
        df = pd.read_csv(url)
        # Filter for games happening in the next 3 days
        df['Date'] = pd.to_datetime(df['Date'])
        today = pd.Timestamp.now()
        upcoming = df[(df['Date'] >= today) & (df['Date'] <= today + pd.Timedelta(days=3))]
        
        matches = []
        for _, row in upcoming.iterrows():
            matches.append({
                "home": row['Home Team'],
                "away": row['Away Team']
            })
            
        with open('upcoming_matches.json', 'w') as f:
            json.dump(matches, f)
        print(f"✅ Created upcoming_matches.json with {len(matches)} games.")
        
    except:
        print("⚠️ Direct fetch failed. Please update upcoming_matches.json manually using the template.")

if __name__ == "__main__":
    generate_fixtures()
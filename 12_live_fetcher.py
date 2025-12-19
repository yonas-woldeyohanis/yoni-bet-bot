import requests
import json

# --- CONFIG ---
# Paste the key from dashboard.api-football.com here
API_KEY = "8f35e7508dae62fb909a6734cca3478b" 
LEAGUE_ID = 41  # League One
SEASON = 2025

def fetch_live_fixtures():
    # Notice the URL is different for direct access
    url = "https://v3.football.api-sports.io/fixtures"
    
    params = {
        "league": LEAGUE_ID,
        "season": SEASON,
        "next": 12
        
    }
    
    headers = {
        "x-apisports-key": API_KEY  # Direct API header
    }

    print("🛰️ Connecting directly to Football-API Servers...")
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    print(data)

    if 'response' not in data or not data['response']:
        print("❌ Error: Could not get fixtures. Check your API key.")
        return

    upcoming = []
    for game in data['response']:
        match_info = {
            "id": game['fixture']['id'],
            "timestamp": game['fixture']['timestamp'],
            "home": game['teams']['home']['name'],
            "away": game['teams']['away']['name']
        }
        upcoming.append(match_info)

    with open('upcoming_matches.json', 'w') as f:
        json.dump(upcoming, f)
    
    print(f"✅ Success! Found {len(upcoming)} upcoming League One matches.")

if __name__ == "__main__":
    fetch_live_fixtures()
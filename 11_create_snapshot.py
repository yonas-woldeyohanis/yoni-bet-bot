import pandas as pd
import json

def create_master_snapshot():
    df = pd.read_csv('master_training_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    latest_stats = {}
    all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    
    for team in all_teams:
        team_row = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].sort_values('Date').iloc[-1]
        
        if team_row['HomeTeam'] == team:
            stats = {
                'goals': team_row['Home_Goals_Avg'], 'corners': team_row['Home_Corners_Avg'],
                'eff': team_row['H_Shot_Eff'], 'btts': team_row['Home_BTTS_Rate']
            }
        else:
            stats = {
                'goals': team_row['Away_Goals_Avg'], 'corners': team_row['Away_Corners_Avg'],
                'eff': team_row['A_Shot_Eff'], 'btts': team_row['Away_BTTS_Rate']
            }
        latest_stats[team] = stats

    with open('team_stats_snapshot.json', 'w') as f:
        json.dump(latest_stats, f)
    print("✅ Master Snapshot Saved.")

if __name__ == "__main__":
    create_master_snapshot()
import pandas as pd
import json

# Settings for a Professional League One Bot
K_FACTOR = 20           # Speed of learning (20 is professional standard)
HOME_ADVANTAGE = 55     # Points added to Home Team (Statistically proven for League One)
INITIAL_ELO = 1500      # Starting point for any new team

def get_expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo():
    # Load our Master Data from Task 2
    df = pd.read_csv('master_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date') # Crucial: Process history in order

    elo_ratings = {}
    
    # Initialize all teams
    all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    for team in all_teams:
        elo_ratings[team] = INITIAL_ELO

    # Create a list to store Elo history for future Machine Learning
    elo_history = []

    print(f"Replaying {len(df)} matches to calculate power levels...")

    for index, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        result = row['FTR']

        # Store ratings BEFORE the match
        row_elo = {
            'Date': row['Date'],
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'Home_Elo_Pre': elo_ratings[home_team],
            'Away_Elo_Pre': elo_ratings[away_team]
        }

        # Calculate Expected outcome (including Home Advantage)
        r_home = elo_ratings[home_team] + HOME_ADVANTAGE
        r_away = elo_ratings[away_team]
        
        expected_home = get_expected_score(r_home, r_away)
        expected_away = get_expected_score(r_away, r_home)

        # Actual result (1=Win, 0.5=Draw, 0=Loss)
        if result == 'H':
            actual_home, actual_away = 1, 0
        elif result == 'A':
            actual_home, actual_away = 0, 1
        else:
            actual_home, actual_away = 0.5, 0.5

        # Update ratings
        elo_ratings[home_team] += K_FACTOR * (actual_home - expected_home)
        elo_ratings[away_team] += K_FACTOR * (actual_away - expected_away)

        elo_history.append(row_elo)

    # Save Current Ratings for prediction phase
    with open('current_elo.json', 'w') as f:
        json.dump(elo_ratings, f)

    # Save detailed history for Task 4 (The ML Training)
    history_df = pd.DataFrame(elo_history)
    history_df.to_csv('elo_history.csv', index=False)

    return elo_ratings

if __name__ == "__main__":
    final_ratings = update_elo()
    
    # Show the Top 10 Teams as of today
    ranking = pd.DataFrame(list(final_ratings.items()), columns=['Team', 'Elo']).sort_values(by='Elo', ascending=False)
    print("\n--- TOP 10 STRONGEST TEAMS (MODEL VERSION 1.0) ---")
    print(ranking.head(10))
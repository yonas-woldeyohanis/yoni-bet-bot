import pandas as pd
import json
import joblib
import numpy as np

# 1. Load the Infrastructure
model_res = joblib.load('model_result.pkl')
model_goals = joblib.load('model_goals.pkl')
model_corners = joblib.load('model_corners.pkl')

with open('current_elo.json', 'r') as f:
    elo_ratings = json.load(f)

# Load training data to get the LATEST rolling stats for each team
df_stats = pd.read_csv('pro_training_data.csv')
df_stats['Date'] = pd.to_datetime(df_stats['Date'])

def get_latest_stats(team_name, is_home=True):
    # Find the last time this team played to get their current rolling averages
    team_data = df_stats[(df_stats['HomeTeam'] == team_name) | (df_stats['AwayTeam'] == team_name)].sort_values('Date').iloc[-1]
    
    if is_home:
        return [team_data['Home_Goals_Avg'], team_data['Home_Corners_Avg'], team_data['Home_Shot_Eff']]
    else:
        return [team_data['Away_Goals_Avg'], team_data['Away_Corners_Avg'], team_data['Away_Shot_Eff']]

def run_dashboard():
    print("\n" + "="*50)
    print("      ⚽ PRO DYNAMICAL COMMAND CENTER (v2.0) ⚽")
    print("="*50)
    
    balance = float(input("Bybit Balance ($): "))
    home = input("Home Team: ").strip()
    away = input("Away Team: ").strip()

    try:
        # Get Elo
        h_elo = elo_ratings[home]
        a_elo = elo_ratings[away]

        # Get Latest Dynamics (Averages)
        h_goals, h_corners, h_eff = get_latest_stats(home, True)
        a_goals, a_corners, a_eff = get_latest_stats(away, False)

        # Build Feature Vector for AI
        features = np.array([[h_elo, a_elo, h_goals, a_goals, h_corners, a_corners, h_eff, a_eff]])

        # 2. RUN ALL BRAINS
        prob_res = model_res.predict_proba(features)[0]   # [Away, Draw, Home]
        prob_goals = model_goals.predict_proba(features)[0] # [Under 2.5, Over 2.5]
        prob_corn = model_corners.predict_proba(features)[0] # [Under 9.5, Over 9.5]

        print(f"\n--- MATCH DYNAMICS: {home} vs {away} ---")
        print(f"📈 Match Outcome:  {home} Win: {prob_res[2]:.1%} | Draw: {prob_res[1]:.1%} | {away} Win: {prob_res[0]:.1%}")
        print(f"🥅 Goal Outlook:   Over 2.5 Goals: {prob_goals[1]:.1%}")
        print(f"🚩 Corner Outlook: Over 9.5 Corners: {prob_corn[1]:.1%}")

        # 3. DYNAMICAL RECOMMENDATION ENGINE
        print("\n--- 🤖 AI STRATEGY ADVICE ---")
        
        # We look for the "Path of Least Resistance" (Highest Probability)
        options = [
            ("Home Win", prob_res[2], "W1"),
            ("Away Win", prob_res[0], "W2"),
            ("Over 2.5 Goals", prob_goals[1], "O2.5"),
            ("Over 9.5 Corners", prob_corn[1], "C9.5")
        ]
        
        # Sort by highest probability
        options.sort(key=lambda x: x[1], reverse=True)
        best_name, best_prob, bet_label = options[0]

        print(f"STRATEGY: The most likely outcome is '{best_name}' ({best_prob:.1%})")
        
        bookie_odds = float(input(f"Enter 1xBet Odds for {bet_label}: "))
        
        # Kelly Criterion for the Best Option
        b = bookie_odds - 1
        edge = (best_prob * bookie_odds) - 1
        
        if edge > 0.02:
            # Use 1/8 Kelly for professional safety
            kelly = ((b * best_prob) - (1 - best_prob)) / b
            bet_amount = (kelly / 8) * balance
            print(f"\n✅ ACTION: Bet ${max(0.20, bet_amount):.2f} on {best_name}")
            print(f"📊 Projected Edge: {edge:.1%}")
        else:
            print("\n❌ NO VALUE: Even though this is likely, the odds are too low.")

    except Exception as e:
        print(f"Error: {e}. Make sure team names match master_data.csv exactly.")

if __name__ == "__main__":
    run_dashboard()
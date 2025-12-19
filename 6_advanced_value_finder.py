import json
import joblib
import pandas as pd
import numpy as np

# Professional Constants
HOME_ADV = 55
DRAW_MARGIN = 85

# Load the AI Brain
model = joblib.load('football_ai_model.pkl')

def run_hybrid_bot():
    # 1. Load the Elo ratings
    with open('current_elo.json', 'r') as f:
        elo = json.load(f)

    print("--- 🤖 ADVANCED AI HYBRID BOT (v1.0) ---")
    balance = float(input("Bybit Balance ($): "))
    home = input("Home Team: ").strip()
    away = input("Away Team: ").strip()

    if home not in elo or away not in elo:
        print("Error: Team name not found in 10-year history.")
        return

    # 2. ASK THE AI (Based on Momentum)
    # Since we don't have tomorrow's live shots, we use the average
    # In a pro setup, you'd use a Live API here.
    home_elo = elo[home]
    away_elo = elo[away]
    
    # We will assume average shots of 4.5 for a "standard" game prediction
    # unless we add a stats-scraper later.
    features = np.array([[home_elo, away_elo, 4.5, 4.2]])
    ai_probs = model.predict_proba(features)[0] 
    
    # ai_probs order: [Away Win, Draw, Home Win]
    pa_ai, pd_ai, ph_ai = ai_probs

    print(f"\nAI Analysis for {home} vs {away}:")
    print(f"Confidence: {ph_ai:.1%} Home | {pd_ai:.1%} Draw | {pa_ai:.1%} Away")

    # 3. COMPARE WITH 1xBet
    bookie_h = float(input(f"Enter 1xBet Odds for {home} (W1): "))
    
    # Calculate Value using AI Probability
    value = (ph_ai * bookie_h) - 1

    if value > 0.05:
        # Kelly Criterion (Quarter Kelly)
        b = bookie_h - 1
        kelly = ((b * ph_ai) - (1 - ph_ai)) / b
        suggested_bet = (kelly / 4) * balance
        
        print(f"\n✅ AI DETECTED VALUE! Edge: {value:.1%}")
        print(f"💰 SUGGESTED BET: ${max(0.20, suggested_bet):.2f}")
    else:
        print(f"\n❌ AI SAYS: NO VALUE. (Wait for better odds)")

if __name__ == "__main__":
    run_hybrid_bot()
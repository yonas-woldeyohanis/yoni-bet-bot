import json
import joblib
import numpy as np

# Load the Brain
model = joblib.load('football_ai_model.pkl')

def run_low_risk_bot():
    with open('current_elo.json', 'r') as f:
        elo = json.load(f)

    print("--- 🛡️ LOW-RISK SAFETY PREDICTOR (v1.1) ---")
    balance = float(input("Bybit Balance ($): "))
    home = input("Home Team: ").strip()
    away = input("Away Team: ").strip()

    if home not in elo or away not in elo:
        print("Error: Team names not recognized.")
        return

    # 1. AI PREDICTION
    home_elo = elo[home]
    away_elo = elo[away]
    # We use conservative shot averages for safety
    features = np.array([[home_elo, away_elo, 4.0, 4.0]])
    
    # AI Probs: [Away Win (0), Draw (1), Home Win (2)]
    probs = model.predict_proba(features)[0]
    pa, pd, ph = probs[0], probs[1], probs[2]

    # 2. CALCULATE DOUBLE CHANCE PROBABILITIES
    # 1X = Home Win + Draw
    # X2 = Away Win + Draw
    p_1x = ph + pd
    p_x2 = pa + pd

    print(f"\n--- 📊 Safety Analysis ---")
    print(f"Direct Win Confidence: {max(ph, pa):.1%}")
    print(f"Double Chance {home} (1X): {p_1x:.1%}")
    print(f"Double Chance {away} (X2): {p_x2:.1%}")

    # 3. LOW RISK CRITERIA
    # We only move forward if one outcome has > 75% probability (Very Safe)
    if p_1x > 0.75:
        target_team = home
        target_prob = p_1x
        bet_type = "Double Chance 1X"
    elif p_x2 > 0.75:
        target_team = away
        target_prob = p_x2
        bet_type = "Double Chance X2"
    else:
        print("\n⚠️ RISK TOO HIGH: No Double Chance exceeds 75% confidence.")
        return

    print(f"\n🛡️ LOW RISK OPPORTUNITY FOUND: {target_team}")
    bookie_odds = float(input(f"Enter 1xBet Odds for {bet_type}: "))
    
    # Value check still matters even for low risk
    if (target_prob * bookie_odds) > 1.02:
        # Use a very small fraction of bankroll for low risk (1/8 Kelly)
        b = bookie_odds - 1
        kelly = ((b * target_prob) - (1 - target_prob)) / b
        suggested_bet = (kelly / 8) * balance
        
        print(f"\n💰 SUGGESTED SAFE BET: ${max(0.20, suggested_bet):.2f} on {bet_type}")
        print(f"Probability of Winning: {target_prob:.1%}")
    else:
        print("\n❌ ODDS TOO LOW: The payout isn't worth the risk.")

if __name__ == "__main__":
    run_low_risk_bot()
import json

# Professional Constants
HOME_ADV = 55
DRAW_MARGIN = 85

def get_probability(rating_home, rating_away):
    diff = (rating_home + HOME_ADV) - rating_away
    win_p = 1 / (1 + 10 ** ((-diff + DRAW_MARGIN) / 400))
    loss_p = 1 / (1 + 10 ** ((diff + DRAW_MARGIN) / 400))
    draw_p = 1 - win_p - loss_p
    return win_p, draw_p, loss_p

def kelly_criterion(prob, odds, bankroll):
    # The Kelly Formula: (bp - q) / b
    # b = decimal odds - 1, p = probability, q = probability of losing
    b = odds - 1
    p = prob
    q = 1 - p
    
    fraction = (b * p - q) / b
    
    # We use "Quarter Kelly" (divide by 4) to be safe/professional
    safe_fraction = fraction / 4
    
    if safe_fraction < 0:
        return 0
    return safe_fraction * bankroll

def run_calculator():
    with open('current_elo.json', 'r') as f:
        elo = json.load(f)

    # Ask for your Bybit balance
    balance = float(input("Enter your current Bybit Wallet Balance ($): "))
    home = input("Enter Home Team: ").strip()
    away = input("Enter Away Team: ").strip()

    if home not in elo or away not in elo:
        print("Error: Team names not found!")
        return

    ph, pd, pa = get_probability(elo[home], elo[away])
    
    print(f"\n--- Probabilities ---")
    print(f"{home}: {ph:.1%} | Draw: {pd:.1%} | {away}: {pa:.1%}")

    bookie_h = float(input(f"Enter 1xBet Odds for {home} (W1): "))
    
    # Calculate Kelly Bet
    suggested_bet = kelly_criterion(ph, bookie_h, balance)

    if suggested_bet > 0:
        print(f"\n✅ VALUE FOUND!")
        print(f"Suggested Bet: ${suggested_bet:.2f}")
        print(f"This is { (suggested_bet/balance):.1%} of your bankroll.")
    else:
        print(f"\n❌ NO VALUE found for {home} at these odds.")

if __name__ == "__main__":
    run_calculator()
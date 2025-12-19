import pandas as pd
import numpy as np

def create_elite_features():
    df = pd.read_csv('master_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Load Elo history
    elo_df = pd.read_csv('elo_history.csv')
    elo_df['Date'] = pd.to_datetime(elo_df['Date'])
    df = pd.merge(df, elo_df, on=['Date', 'HomeTeam', 'AwayTeam'])

    # --- 1. DYNAMICAL GOAL FEATURES ---
    # Both Teams to Score (BTTS)
    df['BTTS_Actual'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
    
    # Rolling BTTS Rate (Last 10 games) - shows if teams play "open" football
    df['Home_BTTS_Rate'] = df.groupby('HomeTeam')['BTTS_Actual'].transform(lambda x: x.rolling(10, closed='left').mean())
    df['Away_BTTS_Rate'] = df.groupby('AwayTeam')['BTTS_Actual'].transform(lambda x: x.rolling(10, closed='left').mean())

    # --- 2. QUANT xG (The "Threat" Level) ---
    # We calculate the probability of a shot becoming a goal for each team
    def calc_conv_rate(goals, shots):
        return (goals.rolling(10).sum() / shots.rolling(10).sum().replace(0, 1)).shift()

    df['H_Conv_Rate'] = df.groupby('HomeTeam').apply(lambda x: calc_conv_rate(x['FTHG'], x['HS'])).reset_index(0, drop=True)
    df['A_Conv_Rate'] = df.groupby('AwayTeam').apply(lambda x: calc_conv_rate(x['FTAG'], x['AS'])).reset_index(0, drop=True)

    # Current xG = Historical Conversion Rate * Shots in this match
    # (For training, we use actual shots; for prediction, we use rolling avg shots)
    df['H_xG'] = df['HS'] * df['H_Conv_Rate']
    df['A_xG'] = df['AS'] * df['A_Conv_Rate']

    # --- 3. TARGETS FOR MULTI-MARKET BETTING ---
    df['Target_Result'] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0})
    df['Target_Over25'] = ( (df['FTHG'] + df['FTAG']) > 2.5 ).astype(int)
    df['Target_BTTS'] = df['BTTS_Actual']
    df['Target_Home_Over15'] = (df['FTHG'] > 1.5).astype(int) # Team specific goals
    df['Target_Away_Over15'] = (df['FTAG'] > 1.5).astype(int)

    df = df.dropna()
    df.to_csv('elite_training_data.csv', index=False)
    print(f"✅ Elite Training Data Created: {len(df)} matches with BTTS & xG.")

if __name__ == "__main__":
    create_elite_features()
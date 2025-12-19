import pandas as pd
import numpy as np

def create_master_features():
    df = pd.read_csv('master_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    elo_df = pd.read_csv('elo_history.csv')
    elo_df['Date'] = pd.to_datetime(elo_df['Date'])
    df = pd.merge(df, elo_df, on=['Date', 'HomeTeam', 'AwayTeam'])

    # 1. GOAL DYNAMICS (Last 10 games)
    df['Home_Goals_Avg'] = df.groupby('HomeTeam')['FTHG'].transform(lambda x: x.rolling(10, closed='left').mean())
    df['Away_Goals_Avg'] = df.groupby('AwayTeam')['FTAG'].transform(lambda x: x.rolling(10, closed='left').mean())
    
    # 2. CORNER DYNAMICS
    df['Home_Corners_Avg'] = df.groupby('HomeTeam')['HC'].transform(lambda x: x.rolling(10, closed='left').mean())
    df['Away_Corners_Avg'] = df.groupby('AwayTeam')['AC'].transform(lambda x: x.rolling(10, closed='left').mean())

    # 3. EFFICIENCY (Goals per Shot)
    df['H_Shot_Eff'] = df['Home_Goals_Avg'] / df.groupby('HomeTeam')['HS'].transform(lambda x: x.rolling(10, closed='left').mean()).replace(0, 1)
    df['A_Shot_Eff'] = df['Away_Goals_Avg'] / df.groupby('AwayTeam')['AS'].transform(lambda x: x.rolling(10, closed='left').mean()).replace(0, 1)

    # 4. BTTS DYNAMICS
    df['BTTS_Actual'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
    df['Home_BTTS_Rate'] = df.groupby('HomeTeam')['BTTS_Actual'].transform(lambda x: x.rolling(10, closed='left').mean())
    df['Away_BTTS_Rate'] = df.groupby('AwayTeam')['BTTS_Actual'].transform(lambda x: x.rolling(10, closed='left').mean())

    # 5. TARGETS
    df['Target_Result'] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0})
    df['Target_Over25'] = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int)
    df['Target_BTTS'] = df['BTTS_Actual']
    df['Target_Corners10'] = ((df['HC'] + df['AC']) > 9.5).astype(int)

    df = df.dropna()
    df.to_csv('master_training_data.csv', index=False)
    print(f"✅ Master Matrix Created: {len(df)} matches with 10 features.")

if __name__ == "__main__":
    create_master_features()
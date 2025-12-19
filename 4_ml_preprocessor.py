import pandas as pd

def create_features():
    # Load our master data
    df = pd.read_csv('master_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Load Elo history we created in Task 3
    elo_df = pd.read_csv('elo_history.csv')
    elo_df['Date'] = pd.to_datetime(elo_df['Date'])
    
    # Merge Elo into our main data
    df = pd.merge(df, elo_df, on=['Date', 'HomeTeam', 'AwayTeam'])

    # SIMPLE ML FEATURE: Rolling Shots on Target (Last 5 games)
    # This teaches the bot "Attacking Momentum"
    print("Calculating Attacking Momentum (Shots on Target)...")
    
    # We calculate the average shots on target for the last 5 home/away games
    df['Home_Shots_Avg'] = df.groupby('HomeTeam')['HST'].transform(lambda x: x.rolling(5, closed='left').mean())
    df['Away_Shots_Avg'] = df.groupby('AwayTeam')['AST'].transform(lambda x: x.rolling(5, closed='left').mean())

    # Create the TARGET (What the bot is trying to predict)
    # 2 = Home Win, 1 = Draw, 0 = Away Win
    def map_res(res):
        if res == 'H': return 2
        if res == 'D': return 1
        return 0
    
    df['Target'] = df['FTR'].apply(map_res)

    # Drop the first few games where we don't have enough "Rolling" data
    df = df.dropna()

    # Save the training data
    df.to_csv('training_data.csv', index=False)
    print(f"Done! {len(df)} matches ready for ML training.")

if __name__ == "__main__":
    create_features()
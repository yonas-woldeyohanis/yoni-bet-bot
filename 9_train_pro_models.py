import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_master_brains():
    df = pd.read_csv('master_training_data.csv')
    
    # THE 10 FEATURE LIST
    features = [
        'Home_Elo_Pre', 'Away_Elo_Pre', 
        'Home_Goals_Avg', 'Away_Goals_Avg',
        'Home_Corners_Avg', 'Away_Corners_Avg',
        'H_Shot_Eff', 'A_Shot_Eff',
        'Home_BTTS_Rate', 'Away_BTTS_Rate'
    ]
    
    targets = {
        'result': 'Target_Result',
        'goals': 'Target_Over25',
        'btts': 'Target_BTTS',
        'corners': 'Target_Corners10'
    }

    for name, target in targets.items():
        model = RandomForestClassifier(n_estimators=200, min_samples_split=10, random_state=1)
        model.fit(df[features], df[target])
        joblib.dump(model, f'model_{name}.pkl')
        print(f"Brain Trained: {name}")

if __name__ == "__main__":
    train_master_brains()
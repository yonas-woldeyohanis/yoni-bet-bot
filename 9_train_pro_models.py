import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_elite_brains():
    df = pd.read_csv('elite_training_data.csv')
    
    # Advanced Feature Set
    features = [
        'Home_Elo_Pre', 'Away_Elo_Pre', 
        'H_Conv_Rate', 'A_Conv_Rate',
        'Home_BTTS_Rate', 'Away_BTTS_Rate'
    ]
    
    targets = {
        'result': 'Target_Result',
        'goals': 'Target_Over25',
        'btts': 'Target_BTTS',
        'home_o15': 'Target_Home_Over15',
        'away_o15': 'Target_Away_Over15',
        'corners': 'Target_Corners10'
    }

    for name, target in targets.items():
        model = RandomForestClassifier(n_estimators=200, random_state=1)
        model.fit(df[features], df[target])
        joblib.dump(model, f'model_{name}.pkl')
        print(f"Brain Trained: {name}")

if __name__ == "__main__":
    train_elite_brains()
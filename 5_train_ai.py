import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib # For saving the brain

def train_brain():
    # 1. Load the data from Task 6
    df = pd.read_csv('training_data.csv')

    # 2. Select our "Predictors" (The facts the scouts look at)
    # We use Elo and the Attacking Momentum we calculated
    predictors = ['Home_Elo_Pre', 'Away_Elo_Pre', 'Home_Shots_Avg', 'Away_Shots_Avg']
    
    # 3. Split the data (Professional standard)
    # We train on the first 80% of matches and "Test" on the last 20% 
    # to see if the bot actually learned or just guessed.
    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    # 4. Initialize the Random Forest
    # n_estimators=100 means we are asking 100 "scouts"
    # min_samples_split helps prevent the bot from over-thinking (overfitting)
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)

    # 5. THE TRAINING (The "Learning" Phase)
    print("AI is studying 10 years of League One history...")
    rf.fit(train[predictors], train['Target'])

    # 6. THE TEST (Checking its homework)
    preds = rf.predict(test[predictors])
    acc = accuracy_score(test['Target'], preds)

    print(f"\n--- Task 7 Complete ---")
    print(f"AI Accuracy Score: {acc:.2%}")
    
    # 7. Save the Brain
    # This saves the model as a file so we don't have to train it every time.
    joblib.dump(rf, 'football_ai_model.pkl')
    print("Brain saved as: football_ai_model.pkl")

if __name__ == "__main__":
    train_brain()
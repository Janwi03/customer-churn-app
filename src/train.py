import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import load_data, preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

def train_and_save_model(data_path, model_path):
    df = load_data(data_path)
    X, y, encoder, scaler = preprocess_data(df, fit=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save model, encoder, scaler
    with open("models/churn_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

if __name__ == "__main__":
    train_and_save_model("data/Telco_customer_churn.xlsx", "models/churn_model.pkl")
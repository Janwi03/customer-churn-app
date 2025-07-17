import pickle
import pandas as pd

def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def predict(model, input_dict):
    input_df = pd.DataFrame([input_dict])
    return model.predict(input_df)[0]

import pandas as pd
import pickle
import os
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report 
import config
from sklearn.preprocessing import LabelEncoder 
import numpy as np 

def train_model():
    print("Training Decision Tree Model...")
    
    df = pd.read_csv(config.DATASET_PATH)
    
    relevant_cols = config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    df = df[relevant_cols].dropna()
    
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    model = DecisionTreeClassifier(random_state=config.RANDOM_STATE, max_depth=10)
    model.fit(X_train, y_train_encoded) 

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    
    print(f"Decision Tree - Accuracy: {accuracy:.4f}")

    
    os.makedirs("models", exist_ok=True)
    
    with open(config.DECISION_TREE_MODEL, "wb") as f:
        pickle.dump(model, f)
        
    with open(config.LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    
    print(f"Model saved to {config.DECISION_TREE_MODEL}")
    print(f"Label Encoder saved to {config.LABEL_ENCODER_PATH}")
    return model, le


def load_model_and_encoder():
    """Helper function to load both model and encoder."""
    if not os.path.exists(config.DECISION_TREE_MODEL) or not os.path.exists(config.LABEL_ENCODER_PATH):
        print("Model or encoder not found. Training new model...")
        train_model()
    
    with open(config.DECISION_TREE_MODEL, "rb") as f:
        model = pickle.load(f)
    with open(config.LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
        
    return model, le


def predict(input_data):
    model, le = load_model_and_encoder()
  
    input_df = pd.DataFrame([input_data], columns=config.FEATURE_COLUMNS)

    prediction_numeric = model.predict(input_df)[0]

    prediction_name = le.inverse_transform([int(prediction_numeric)])[0]
    
    return prediction_name 


if __name__ == "__main__":
    train_model()

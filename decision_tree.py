import pandas as pd
import pickle
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import config


def train_model():
    print("Training Decision Tree Model...")
    
    df = pd.read_csv(config.DATASET_PATH)
    
    relevant_cols = config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    df = df[relevant_cols].dropna()
    
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    model = DecisionTreeRegressor(random_state=config.RANDOM_STATE, max_depth=10)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Decision Tree - MSE: {mse:.4f}, R²: {r2:.4f}")
    
    os.makedirs("models", exist_ok=True)
    with open(config.DECISION_TREE_MODEL, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {config.DECISION_TREE_MODEL}")
    return model


def load_model():
    if not os.path.exists(config.DECISION_TREE_MODEL):
        print("Model not found. Training new model...")
        return train_model()
    
    with open(config.DECISION_TREE_MODEL, "rb") as f:
        model = pickle.load(f)
    return model


def predict(input_data):
    model = load_model()
  
    input_df = pd.DataFrame([input_data], columns=config.FEATURE_COLUMNS)

    prediction = model.predict(input_df)[0]
    
    return prediction


if __name__ == "__main__":
    train_model()

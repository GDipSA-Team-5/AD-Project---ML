"""
Final ML Model for Waste Bin Fill-Level Prediction

This script contains the linear regression model that predicts
the average daily fill growth for the next collection cycle.

The code was moved from Jupyter notebooks to a separate .py file to:
- Make it easier to test
- Allow SonarQube to scan the code
- Keep notebooks for exploration and this file for the final model
"""

from datasets import load_dataset
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os
import joblib

def load_data():
    print("Loading dataset...")

    dataset = load_dataset("SA61team5/AD-tableC")
    df = pd.DataFrame(dataset["train"])
    
    print(f"Loaded {len(df)} records")
    return df

def prepare_features(df):
    df = df.copy()
    df["start_timestamp"] = pd.to_datetime(df["start_timestamp"])

    # Use time-based split: 75% train, 25% test
    split_date = df["start_timestamp"].quantile(0.75)

    train_df = df[df["start_timestamp"] <= split_date]
    test_df = df[df["start_timestamp"] > split_date]

    features = [
        "ContainerID",
        "collection_fill_percentage",
        "cycle_start_month",
        "cycle_duration_days",
        "avg_daily_fill_growth",
    ]

    target = "next_cycle_avg_daily_fill_growth"

    X_train_raw = train_df[features]
    X_test_raw = test_df[features]

    # One-hot encode ContainerID 
    X_train = pd.get_dummies(X_train_raw, columns=["ContainerID"], drop_first=True)
    X_test = pd.get_dummies(X_test_raw, columns=["ContainerID"], drop_first=True)

    # Make sure train and test have same columns
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    y_train = train_df[target]
    y_test = test_df[target]

    return X_train, X_test, y_train, y_test

def train_and_evaluate(X_train, X_test, y_train, y_test):
    print("Training model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }
    
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
    return model, metrics

def save_artifacts(model, feature_columns, output_dir="."):
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save model and features
    feature_path = os.path.join(output_dir, "feature_columns.pkl")
    model_path = os.path.join(output_dir, "model.pkl")
    
    joblib.dump(feature_columns, feature_path)
    joblib.dump(model, model_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Features saved to: {feature_path}")


def main(save_model=False, output_dir="."):
    print("=" * 50)
    print("Starting ML Pipeline")
    print("=" * 50)
    
    # 1: Load data
    df = load_data()
    
    # 2: Prepare features
    X_train, X_test, y_train, y_test = prepare_features(df)
    
    # 3: Train and evaluate
    model, metrics = train_and_evaluate(X_train, X_test, y_train, y_test)

    # 4: Save model if requested
    if save_model:
        save_artifacts(model, X_train.columns.tolist(), output_dir)
    
    print("=" * 50)
    print("Pipeline completed!")
    print("=" * 50)
    
    return metrics

if __name__ == "__main__":
    main(save_model=True)

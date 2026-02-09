import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import tempfile
import shutil
import os

from final_LR_Model import (
    load_data,
    prepare_features,
    train_and_evaluate,
    save_artifacts,
    main
)


def test_load_data():
    """Test that we can load the dataset."""
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    print("Load data test passed")


def test_load_data_has_columns():
    """Test that the dataset has the columns we need."""
    df = load_data()
    required_columns = [
        "ContainerID",
        "collection_fill_percentage",
        "cycle_start_month",
        "cycle_duration_days",
        "avg_daily_fill_growth",
        "next_cycle_avg_daily_fill_growth",
        "start_timestamp"
    ]
    for col in required_columns:
        assert col in df.columns
    print("Column check test passed")


def test_prepare_features():
    """Test that prepare_features returns the right data."""
    # Create some sample data
    df = pd.DataFrame({
        "ContainerID": [1, 1, 2, 2, 3, 3, 4, 4],
        "collection_fill_percentage": [50, 60, 70, 80, 45, 55, 65, 75],
        "cycle_start_month": [1, 2, 3, 4, 5, 6, 7, 8],
        "cycle_duration_days": [10, 12, 8, 15, 9, 11, 13, 14],
        "avg_daily_fill_growth": [5.0, 5.0, 8.75, 5.33, 5.0, 5.0, 5.0, 5.36],
        "next_cycle_avg_daily_fill_growth": [6.0, 7.0, 5.5, 4.5, 6.1, 5.9, 5.8, 5.4],
        "start_timestamp": pd.date_range("2024-01-01", periods=8, freq="D")
    })
    
    X_train, X_test, y_train, y_test = prepare_features(df)
    
    # Check that we got DataFrames and Series
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    
    # Check sizes
    assert len(X_train) + len(X_test) == len(df)
    print("Prepare features test passed")


def test_train_and_evaluate():
    """Test that the model trains and returns metrics."""
    # Create simple test data
    np.random.seed(42)
    X_train = pd.DataFrame({
        "feature1": np.random.rand(50),
        "feature2": np.random.rand(50)
    })
    y_train = pd.Series(2 * X_train["feature1"] + 3 * X_train["feature2"])
    
    X_test = pd.DataFrame({
        "feature1": np.random.rand(20),
        "feature2": np.random.rand(20)
    })
    y_test = pd.Series(2 * X_test["feature1"] + 3 * X_test["feature2"])
    
    model, metrics = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Check that we got a model and metrics
    assert isinstance(model, LinearRegression)
    assert isinstance(metrics, dict)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    print("Train and evaluate test passed")


def test_save_artifacts():
    """Test that we can save the model files."""
    temp_dir = tempfile.mkdtemp()
    try:
        model = LinearRegression()
        features = ["feature1", "feature2", "feature3"]
        
        save_artifacts(model, features, temp_dir)
        
        # Check that files were created
        assert os.path.exists(os.path.join(temp_dir, "model.pkl"))
        assert os.path.exists(os.path.join(temp_dir, "feature_columns.pkl"))
        print("Save artifacts test passed")
    finally:
        shutil.rmtree(temp_dir)


def test_main():
    """Test that the main pipeline runs without errors."""
    metrics = main(save_model=False)
    
    assert isinstance(metrics, dict)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    print("Main pipeline test passed")


if __name__ == "__main__":
    # Run tests
    print("Running tests...")
    print()
    test_load_data()
    test_load_data_has_columns()
    test_prepare_features()
    test_train_and_evaluate()
    test_save_artifacts()
    test_main()
    print()
    print("All tests passed!")

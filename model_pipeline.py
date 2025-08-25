# model_pipeline.py
"""
Core logic for the model pipeline: data loading, preprocessing, training,
evaluation, and prediction.
"""
from typing import Tuple, List

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import shutil

import config

def validate_training_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates the structure and data types of the uploaded training DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to validate.

    Returns:
        A tuple containing a boolean (True if valid) and a list of error messages.
    """
    errors = []
    required_columns = config.FEATURES + [config.TARGET]

    # 1. Check for missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        return False, errors  # Return early if columns are missing

    # 2. Check data types
    for col in config.NUMERICAL_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' must be a numeric type (e.g., integer or float).")

    for col in config.CATEGORICAL_FEATURES:
        # We check for 'object' dtype which is what Pandas uses for strings
        if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col]):
            errors.append(f"Column '{col}' must be a text/categorical type.")

    if not pd.api.types.is_numeric_dtype(df[config.TARGET]):
        errors.append(f"Target column '{config.TARGET}' must be numeric.")

    is_valid = len(errors) == 0
    return is_valid, errors

def load_data(filepath: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    return pd.read_csv(filepath)


def create_pipeline(model_params: dict) -> Pipeline:
    """
    Creates the scikit-learn pipeline with configurable model parameters.
    """
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, config.NUMERICAL_FEATURES),
            ('cat', categorical_transformer, config.CATEGORICAL_FEATURES)
        ])

    # The classifier now uses parameters passed from the UI
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, **model_params))
    ])
    return pipeline


def train_model(data_path: str, model_params: dict) -> str:
    """
    Trains a new model using the given data and parameters,
    and saves it to the temporary 'candidate' path.
    """
    print("Starting model training...")
    data = load_data(data_path)
    X_train = data[config.FEATURES]
    y_train = data[config.TARGET]

    pipeline = create_pipeline(model_params)
    pipeline.fit(X_train, y_train)

    os.makedirs(os.path.dirname(config.TEMP_MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, config.TEMP_MODEL_PATH)

    print(f"Candidate model trained and saved to {config.TEMP_MODEL_PATH}")
    return config.TEMP_MODEL_PATH


def evaluate_model(model_path: str) -> dict:
    """
    Evaluates a model stored at a specific path using the default test set.
    """
    print(f"Evaluating model at {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}.")

    pipeline = joblib.load(model_path)
    X_test = load_data(config.TEST_DATA_PATH)[config.FEATURES]
    y_test_df = load_data(config.SUBMISSION_DATA_PATH)
    y_test = y_test_df[config.TARGET]

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"Evaluation complete. Accuracy: {accuracy:.4f}")
    return {"accuracy": accuracy, "classification_report": report}


def promote_model_to_production() -> str:
    """
    Promotes the candidate model to production by overwriting the main model file.
    """
    if not os.path.exists(config.TEMP_MODEL_PATH):
        raise FileNotFoundError("No candidate model found to promote.")

    shutil.copy(config.TEMP_MODEL_PATH, config.MODEL_PATH)
    os.remove(config.TEMP_MODEL_PATH)

    success_message = f"✅ Model promoted to production at `{config.MODEL_PATH}`"
    print(success_message)
    return success_message


def run_prediction(input_data: pd.DataFrame) -> tuple[int, float]:
    """
    Runs a prediction using the current production model.
    """
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Production model not found at {config.MODEL_PATH}. Please train a model first.")

    pipeline = joblib.load(config.MODEL_PATH)
    prediction = pipeline.predict(input_data)[0]
    probabilities = pipeline.predict_proba(input_data)
    survival_probability = probabilities[0][1]

    return prediction, survival_probability
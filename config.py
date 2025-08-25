# config.py
"""
Configuration file for the Titanic Survival Prediction project.
Stores constants like file paths, feature names, and model parameters.
"""

# --- File Paths ---
TRAIN_DATA_PATH = "data/train.csv"
TEST_DATA_PATH = "data/test.csv"
SUBMISSION_DATA_PATH = "data/gender_submission.csv"

# The 'production' model that the app uses for predictions
MODEL_PATH = "models/titanic_logistic_regression.joblib"
# A temporary path to save a newly trained 'candidate' model before it's promoted
TEMP_MODEL_PATH = "models/temp_candidate_model.joblib"

# --- Feature Engineering & Model Configuration ---
TARGET = 'Survived'
FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
CATEGORICAL_FEATURES = ['Sex', 'Embarked']
NUMERICAL_FEATURES = ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass']
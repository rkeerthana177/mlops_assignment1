import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os
from pathlib import Path
from datetime import datetime

# Set the tracking URI to the current directory
tracking_uri = Path(os.getcwd()) / "mlruns"
mlflow.set_tracking_uri(tracking_uri.as_uri())

# Load the custom dataset
data = pd.read_csv('age_prediction_dataset.csv')

# One-hot encode categorical features
data = pd.get_dummies(data, columns=['Age_group', 'Gender', 'PAQ605', 'Diabetic or not', "Respondent's Oral"])

X = data.drop(['ID', 'Age'], axis=1)  # Drop ID and target column
y = data['Age']  # Target column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train and log the model
def train_and_log_model(n_estimators, max_depth, learning_rate, epochs):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
    # Set the run name
    run_name = f"RandomForest_AgePredictor_{timestamp}"
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)
        
        # Train the model
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict and evaluate the model
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        
        # Log the model and metrics
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_metric("mse", mse)

        print(f"Run with n_estimators={n_estimators}, max_depth={max_depth}, mse={mse}")

# Run experiments with different values of n_estimators, max_depth, learning_rate, and epochs
train_and_log_model(n_estimators=100, max_depth=5, learning_rate=0.01, epochs=200)
train_and_log_model(n_estimators=200, max_depth=10, learning_rate=0.1, epochs=20)
train_and_log_model(n_estimators=300, max_depth=15, learning_rate=0.5, epochs=100)

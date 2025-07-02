import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from datetime import datetime

# Function to load processed data
def load_processed_data(file_path):
    return pd.read_csv(file_path)

# Function to evaluate model performance
def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }
    print(f"{model_name} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    return metrics

# Function to train and track models
def train_models(X_train, X_test, y_train, y_test, experiment_name="Credit_Risk_Model"):
    # Set up MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Define models and hyperparameter grids
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42),
            'params': {
                'C': [0.01, 0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
        }
    }
    
    best_model = None
    best_roc_auc = 0
    best_model_name = ""
    
    for name, config in models.items():
        with mlflow.start_run(run_name=f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params(config['params'])
            
            # Perform GridSearchCV
            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['params'],
                cv=5,
                scoring='roc_auc',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Get best model
            model = grid_search.best_estimator_
            
            # Predict on test set
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Evaluate model
            metrics = evaluate_model(y_test, y_pred, y_pred_proba, name)
            
            # Log metrics to MLflow
            mlflow.log_metrics(metrics)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(model, f"{name}_model")
            
            # Register model if it has the best ROC-AUC
            if metrics['roc_auc'] > best_roc_auc:
                best_roc_auc = metrics['roc_auc']
                best_model = model
                best_model_name = name
            
            # Log best parameters
            mlflow.log_params(grid_search.best_params_)
    
    # Register the best model in MLflow Model Registry
    if best_model:
        with mlflow.start_run(run_name=f"Best_Model_{best_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.sklearn.log_model(
                best_model,
                f"best_{best_model_name}_model",
                registered_model_name=f"CreditRisk_{best_model_name}"
            )
            mlflow.log_metric("best_roc_auc", best_roc_auc)
            print(f"Best model ({best_model_name}) registered with ROC-AUC: {best_roc_auc:.4f}")

# Main function to run training
def main(input_path, test_size=0.2, random_state=42):
    # Load processed data
    df = load_processed_data(input_path)
    
    # Separate features and target
    X = df.drop(columns=['CustomerId', 'is_high_risk'])
    y = df['is_high_risk']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Train models and track with MLflow
    train_models(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    input_file = '../data/processed/processed_data.csv'  # Update path as needed
    main(input_file)
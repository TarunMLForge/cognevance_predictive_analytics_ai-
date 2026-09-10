import pandas as pd
import numpy as np
import os
import joblib
import json

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def build_preprocessing_pipeline(numerical_cols, categorical_cols):
    """Creates a ColumnTransformer to handle scaling and encoding."""
    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_cols),
            ('cat', cat_transformer, categorical_cols)
        ])
    return preprocessor

def evaluate_model(y_true, y_pred, y_prob):
    """Calculates classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }

def train_and_compare():
    print("Loading split data...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    
    X_train = train_df.drop(columns=['default'])
    y_train = train_df['default']
    
    X_val = val_df.drop(columns=['default'])
    y_val = val_df['default']
    
    # Define columns
    categorical_cols = ['sex', 'education', 'marriage']
    numerical_cols = [col for col in X_train.columns if col not in categorical_cols]
    
    preprocessor = build_preprocessing_pipeline(numerical_cols, categorical_cols)
    
    models = {
        'Baseline_LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42)
    }
    
    results = []
    best_roc = 0
    best_model_name = ""
    best_pipeline = None
    
    print("Training models...")
    for name, model in models.items():
        print(f"--> Training {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_val)
        y_prob = pipeline.predict_proba(X_val)[:, 1]
        
        metrics = evaluate_model(y_val, y_pred, y_prob)
        metrics['model'] = name
        results.append(metrics)
        
        # Track best model based on ROC-AUC
        if metrics['roc_auc'] > best_roc:
            best_roc = metrics['roc_auc']
            best_model_name = name
            best_pipeline = pipeline
            
    # Save comparison report
    results_df = pd.DataFrame(results).set_index('model')
    os.makedirs("outputs/metrics", exist_ok=True)
    results_df.to_csv("outputs/metrics/model_comparison.csv")
    print("\nModel Comparison Table:")
    print(results_df)
    
    # Save best model
    os.makedirs("models", exist_ok=True)
    model_path = "models/best_model_pipeline.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"\nBest model selected: {best_model_name} (ROC-AUC: {best_roc:.4f})")
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_compare()

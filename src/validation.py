import pandas as pd
import numpy as pd_np
import os

def validate_data(filepath="data/raw/credit_card_defaults.csv", report_path="outputs/reports/data_validation_report.txt"):
    """
    Performs data validation on the raw dataset and generates a summary report.
    """
    print(f"Loading data from {filepath} for validation...")
    df = pd.read_csv(filepath)
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("=========================================\n")
        f.write("DATA VALIDATION REPORT\n")
        f.write("=========================================\n\n")
        
        # Shape
        f.write(f"1. Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")
        
        # Data types
        f.write("2. Data Types:\n")
        f.write(df.dtypes.to_string() + "\n\n")
        
        # Missing values
        f.write("3. Missing Values:\n")
        missing = df.isnull().sum()
        f.write(missing[missing > 0].to_string() if missing.sum() > 0 else "No missing values found.\n")
        f.write("\n\n")
        
        # Duplicate rows
        duplicates = df.duplicated().sum()
        f.write(f"4. Duplicate Rows: {duplicates}\n\n")
        
        # Target Distribution
        target_col = 'Y' if 'Y' in df.columns else 'default.payment.next.month'
        if target_col not in df.columns:
            target_col = df.columns[-1] # Fallback to last column
            
        f.write(f"5. Target Distribution ({target_col}):\n")
        f.write(df[target_col].value_counts().to_string() + "\n")
        f.write("Normalized:\n")
        f.write(df[target_col].value_counts(normalize=True).to_string() + "\n\n")
        
        # Unique Values in Categorical/Discrete features
        f.write("6. Unique Values in Categorical Features (SEX, EDUCATION, MARRIAGE):\n")
        for col in ['X2', 'X3', 'X4', 'SEX', 'EDUCATION', 'MARRIAGE']:
            if col in df.columns:
                f.write(f"--- {col} ---\n")
                f.write(df[col].value_counts().to_string() + "\n")
        f.write("\n")
        
        # Basic numerical statistics (to check for outliers/impossible values)
        f.write("7. Numerical Ranges & Outlier Check (Summary Stats):\n")
        f.write(df.describe().T[['min', 'max', 'mean', 'std']].to_string() + "\n\n")
        
    print(f"Validation complete. Report saved to {report_path}")
    
    # Return basic validation flags that might break pipeline
    return {
        "missing_values": int(missing.sum()),
        "duplicates": int(duplicates),
        "imbalance_ratio": df[target_col].value_counts(normalize=True).min()
    }

if __name__ == "__main__":
    validate_data()

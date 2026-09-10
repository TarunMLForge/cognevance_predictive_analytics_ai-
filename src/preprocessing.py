import pandas as pd
import os

def clean_data(input_path="data/raw/credit_card_defaults.csv", output_path="data/processed/cleaned_credit_data.csv"):
    """
    Performs data cleaning: renaming columns, handling duplicates, and treating rare categories.
    """
    print("Loading raw data...")
    df = pd.read_csv(input_path)
    
    # 1. Rename columns for readability
    rename_dict = {
        'X1': 'limit_balance',
        'X2': 'sex',
        'X3': 'education',
        'X4': 'marriage',
        'X5': 'age',
        'X6': 'pay_1',  # Sep
        'X7': 'pay_2',  # Aug
        'X8': 'pay_3',  # Jul
        'X9': 'pay_4',  # Jun
        'X10': 'pay_5', # May
        'X11': 'pay_6', # Apr
        'X12': 'bill_amt1',
        'X13': 'bill_amt2',
        'X14': 'bill_amt3',
        'X15': 'bill_amt4',
        'X16': 'bill_amt5',
        'X17': 'bill_amt6',
        'X18': 'pay_amt1',
        'X19': 'pay_amt2',
        'X20': 'pay_amt3',
        'X21': 'pay_amt4',
        'X22': 'pay_amt5',
        'X23': 'pay_amt6',
        'Y': 'default'
    }
    df = df.rename(columns=rename_dict)
    
    # 2. Drop duplicates
    initial_shape = df.shape[0]
    df = df.drop_duplicates()
    print(f"Dropped {initial_shape - df.shape[0]} duplicate rows.")
    
    # 3. Handle rare/undocumented categories
    # Education: 1=grad school, 2=university, 3=high school, 4=others. Values 0, 5, 6 are undocumented -> group into 4.
    df['education'] = df['education'].replace([0, 5, 6], 4)
    
    # Marriage: 1=married, 2=single, 3=others. Value 0 is undocumented -> group into 3.
    df['marriage'] = df['marriage'].replace(0, 3)
    
    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}. Final shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    clean_data()

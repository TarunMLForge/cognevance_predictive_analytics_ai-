import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

def fetch_and_save_data(output_path="data/raw/credit_card_defaults.csv"):
    """
    Fetches the Default of Credit Card Clients dataset from UCI and saves it to a CSV.
    """
    print("Fetching dataset from UCI ML Repository...")
    # fetch dataset 
    default_of_credit_card_clients = fetch_ucirepo(id=350) 
      
    # data (as pandas dataframes) 
    X = default_of_credit_card_clients.data.features 
    y = default_of_credit_card_clients.data.targets 
    
    # Combine X and y
    df = pd.concat([X, y], axis=1)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    fetch_and_save_data()

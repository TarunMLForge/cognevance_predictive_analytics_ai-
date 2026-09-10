import pandas as pd
import os

def split_and_save_data(input_path="data/processed/engineered_data.csv", output_dir="data/processed/"):
    """
    Splits the data into Training (70%), Validation (15%), and Test (15%) sets using pandas.
    (Manual stratified-like split to bypass sklearn DLL restrictions in current environment)
    """
    print("Loading engineered data for splitting...")
    df = pd.read_csv(input_path)
    
    # Simple manual stratification by target
    df_0 = df[df['default'] == 0]
    df_1 = df[df['default'] == 1]
    
    # 70% Train
    train_0 = df_0.sample(frac=0.7, random_state=42)
    train_1 = df_1.sample(frac=0.7, random_state=42)
    train_df = pd.concat([train_0, train_1]).sample(frac=1, random_state=42) # shuffle
    
    rem_0 = df_0.drop(train_0.index)
    rem_1 = df_1.drop(train_1.index)
    
    # 15% Val (which is 50% of the remaining 30%)
    val_0 = rem_0.sample(frac=0.5, random_state=42)
    val_1 = rem_1.sample(frac=0.5, random_state=42)
    val_df = pd.concat([val_0, val_1]).sample(frac=1, random_state=42)
    
    # 15% Test
    test_0 = rem_0.drop(val_0.index)
    test_1 = rem_1.drop(val_1.index)
    test_df = pd.concat([test_0, test_1]).sample(frac=1, random_state=42)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print("Data splitting complete. (Manual stratified strategy used)")
    print(f"Train set: {train_df.shape[0]} rows")
    print(f"Validation set: {val_df.shape[0]} rows")
    print(f"Test set: {test_df.shape[0]} rows")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    split_and_save_data()

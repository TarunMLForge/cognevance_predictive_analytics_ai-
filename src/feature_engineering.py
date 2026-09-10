import pandas as pd
import os

def create_features(input_path="data/processed/cleaned_credit_data.csv", output_path="data/processed/engineered_data.csv"):
    """
    Engineers new features: Utilization Ratio, Payment Ratio, and Average Delay.
    """
    print(f"Loading cleaned data from {input_path} for feature engineering...")
    df = pd.read_csv(input_path)
    
    # Feature 1: Credit Utilization Ratio (Most recent month)
    # Why it matters: High utilization indicates financial stress.
    df['utilization_ratio'] = df['bill_amt1'] / (df['limit_balance'] + 1e-5)
    
    # Feature 2: Payment Ratio (Most recent month)
    # Why it matters: Paying only a small fraction of the bill indicates inability to clear debt.
    # We use bill_amt2 because pay_amt1 goes towards bill_amt2.
    df['payment_ratio_1'] = df['pay_amt1'] / (df['bill_amt2'].replace(0, 1e-5).abs() + 1e-5)
    
    # Feature 3: Average Delay
    # Why it matters: Consistent delays over 6 months strongly signal default risk.
    # pay_1 to pay_6 represent repayment statuses. Values > 0 are months delayed.
    pay_cols = ['pay_1', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
    # Consider only actual delays (values > 0)
    delay_df = df[pay_cols].clip(lower=0)
    df['avg_delay'] = delay_df.mean(axis=1)
    
    # Feature 4: Max Delay
    df['max_delay'] = delay_df.max(axis=1)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Feature engineering complete. Added {df.shape[1] - 24} new features.")
    print(f"Engineered data saved to {output_path}. Final shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    create_features()

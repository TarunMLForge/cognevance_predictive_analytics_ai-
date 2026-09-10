import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(filepath="data/processed/cleaned_credit_data.csv", output_dir="outputs/figures/"):
    print("Loading cleaned data for EDA...")
    df = pd.read_csv(filepath)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plotting style
    sns.set_theme(style="whitegrid")
    
    # 1. Target Distribution
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x='default', data=df, palette='viridis')
    plt.title('Target Distribution: Default vs No Default')
    plt.xlabel('Default (0=No, 1=Yes)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_target_distribution.png'))
    plt.close()
    
    # 2. Limit Balance Distribution by Default
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='limit_balance', hue='default', bins=30, kde=True, palette='viridis')
    plt.title('Credit Limit Balance Distribution by Default Status')
    plt.xlabel('Limit Balance')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_limit_balance_distribution.png'))
    plt.close()
    
    # 3. Categorical Comparisons (Sex, Education, Marriage)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.countplot(x='sex', hue='default', data=df, ax=axes[0], palette='muted')
    axes[0].set_title('Default by Sex (1=M, 2=F)')
    
    sns.countplot(x='education', hue='default', data=df, ax=axes[1], palette='muted')
    axes[1].set_title('Default by Education (1=Grad, 2=Uni, 3=HS, 4=Other)')
    
    sns.countplot(x='marriage', hue='default', data=df, ax=axes[2], palette='muted')
    axes[2].set_title('Default by Marriage (1=Married, 2=Single, 3=Other)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_categorical_comparisons.png'))
    plt.close()
    
    # 4. Correlation Matrix of Numerical Features
    plt.figure(figsize=(12, 10))
    # Select subset of numeric features for readability (limit_balance, age, bill_amts, pay_amts)
    cols_for_corr = ['limit_balance', 'age'] + [f'bill_amt{i}' for i in range(1, 7)] + [f'pay_amt{i}' for i in range(1, 7)] + ['default']
    corr = df[cols_for_corr].corr()
    sns.heatmap(corr, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Continuous Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_correlation_matrix.png'))
    plt.close()
    
    # 5. Payment Status (pay_1) vs Default
    plt.figure(figsize=(8, 5))
    sns.countplot(x='pay_1', hue='default', data=df, palette='Set2')
    plt.title('Most Recent Repayment Status vs Default (Higher = Delayed)')
    plt.xlabel('Repayment Status in September (-1=Duly, 1=1mo delay, 2=2mo delay...)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_recent_payment_status.png'))
    plt.close()

    print("EDA charts successfully generated and saved to", output_dir)

if __name__ == "__main__":
    run_eda()

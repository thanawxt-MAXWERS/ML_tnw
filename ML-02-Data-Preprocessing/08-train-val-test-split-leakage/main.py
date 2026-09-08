import pandas as pd
from sklearn.model_selection import train_test_split
import os

def data_split():
    print("=== Train-Test Split ===")
    res_path = os.path.join(os.path.dirname(__file__), '..', 'results.csv')
    df = pd.read_csv(res_path)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training set size: {train_df.shape}")
    print(f"Testing set size: {test_df.shape}")

if __name__ == "__main__":
    data_split()
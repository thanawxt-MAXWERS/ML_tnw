import pandas as pd
import os

def handle_duplicates():
    print("=== Part 3: Duplicate Removal ===")
    res_path = os.path.join(os.path.dirname(__file__), '..', 'results.csv')
    df = pd.read_csv(res_path)
    
    print(f"Total Rows before: {len(df)}")
    print(f"Duplicates found: {df.duplicated().sum()}")
    
    df_cleaned = df.drop_duplicates()
    print(f"Total Rows after dropping duplicates: {len(df_cleaned)}")

if __name__ == "__main__":
    handle_duplicates()
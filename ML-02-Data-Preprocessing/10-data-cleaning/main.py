import pandas as pd
import numpy as np
import os

def clean_data():
    print("=== Part 3: Data Cleaning & Type Conversion ===")
    res_path = os.path.join(os.path.dirname(__file__), '..', 'results.csv')
    df = pd.read_csv(res_path)
    
    print("--- ก่อนทำความสะอาด (สังเกตคอลัมน์ position) ---")
    print(df[['resultId', 'position']].head(10))
    
    # จัดการ Inconsistent data (แปลง \N เป็น NaN และเปลี่ยนชนิดเป็น float)
    df['position'] = df['position'].replace(r'\\N', np.nan, regex=True)
    df['position'] = pd.to_numeric(df['position'], errors='coerce')
    
    print("\n--- หลังทำความสะอาด (พร้อมนำไปใช้) ---")
    print(df[['resultId', 'position']].head(10))
    print(f"\nData Type ของ position เปลี่ยนเป็น: {df['position'].dtype}")

if __name__ == "__main__":
    clean_data()
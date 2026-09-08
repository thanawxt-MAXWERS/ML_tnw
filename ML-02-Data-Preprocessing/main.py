import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_lab1_and_lab2():
    print("Loading F1 Datasets...")
    # ตรวจสอบไฟล์
    if not os.path.exists('results.csv') or not os.path.exists('drivers.csv'):
        print("Error: ไม่พบไฟล์ results.csv หรือ drivers.csv")
        return

    # โหลดและรวมตาราง
    df_res = pd.read_csv('results.csv')
    df_dri = pd.read_csv('drivers.csv')
    df = pd.merge(df_res, df_dri, on='driverId', how='left')
    
    # แปลง \N เป็น NaN (ค่าว่าง)
    df.replace(r'\\N', np.nan, regex=True, inplace=True)
    
    print("\n========== LAB 1: Dataset Exploration ==========")
    print(f"Shape: {df.shape}")
    print("\n--- Data Types (Top 10) ---")
    print(df.dtypes.head(10))
    print("\n--- Summary Statistics ---")
    print(df.describe())
    print("\n--- Missing Values (Top 10) ---")
    print(df.isnull().sum().head(10))
    print("\n--- Duplicate Records ---")
    print(f"Total Duplicates: {df.duplicated().sum()}")
    print("\n--- Class Distribution (Nationality) ---")
    print(df['nationality'].value_counts().head())

    print("\n========== LAB 2: Data Visualization ==========")
    # Histogram ดูการกระจายตัวของคะแนน
    plt.figure(figsize=(8, 5))
    df['points'] = pd.to_numeric(df['points'], errors='coerce')
    sns.histplot(df['points'].dropna(), bins=20, kde=True, color='purple')
    plt.title('Distribution of F1 Points')
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64']).dropna()
    if not numeric_df.empty:
        sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.show()

if __name__ == "__main__":
    run_lab1_and_lab2()
import pandas as pd
import numpy as np
import os

def data_transformation():
    print("=== Data Transformation: Log Transform ===")
    res_path = os.path.join(os.path.dirname(__file__), '..', 'results.csv')
    df = pd.read_csv(res_path)
    
    # ใช้ Log transform กับจำนวนรอบ (laps)
    df['laps_log'] = np.log1p(df['laps'])
    print(df[['laps', 'laps_log']].head())

if __name__ == "__main__":
    data_transformation()
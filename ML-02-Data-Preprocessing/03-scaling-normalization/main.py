import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def run_scaling():
    print("=== Feature Scaling ===")
    res_path = os.path.join(os.path.dirname(__file__), '..', 'results.csv')
    df = pd.read_csv(res_path)
    
    # ปรับสเกลคะแนน (points) ให้อยู่ระหว่าง 0-1
    scaler = MinMaxScaler()
    df['points_scaled'] = scaler.fit_transform(df[['points']])
    print(df[['points', 'points_scaled']].head())

if __name__ == "__main__":
    run_scaling()
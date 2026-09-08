import pandas as pd
import os

def feature_engineering():
    print("=== Feature Engineering: Binning ===")
    res_path = os.path.join(os.path.dirname(__file__), '..', 'results.csv')
    df = pd.read_csv(res_path)
    
    # แบ่งกลุ่มคะแนน (Points Binning)
    bins = [-1, 0, 10, 30]
    labels = ['Zero Points', 'Low Points', 'High Points']
    df['points_category'] = pd.cut(df['points'], bins=bins, labels=labels)
    
    print(df[['points', 'points_category']].head(10))

if __name__ == "__main__":
    feature_engineering()
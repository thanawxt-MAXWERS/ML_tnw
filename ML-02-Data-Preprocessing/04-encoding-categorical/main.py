import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def run_encoding():
    print("=== Part 4: Feature Engineering (Encoding) ===")
    dri_path = os.path.join(os.path.dirname(__file__), '..', 'drivers.csv')
    df = pd.read_csv(dri_path)
    
    # ใช้สัญชาตินักแข่ง (nationality)
    le = LabelEncoder()
    df['nationality_label'] = le.fit_transform(df['nationality'])
    
    print("\n--- Label Encoding ---")
    print(df[['nationality', 'nationality_label']].head())
    
    print("\n--- One-Hot Encoding ---")
    df_onehot = pd.get_dummies(df[['driverRef', 'nationality']], columns=['nationality'])
    print(df_onehot.head())

if __name__ == "__main__":
    run_encoding()
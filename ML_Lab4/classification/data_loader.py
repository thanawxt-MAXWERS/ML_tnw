import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath, target_col):
    df = pd.read_csv(filepath)
    X = df.drop(columns=[target_col]).select_dtypes(include=['number'])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    test_data_original = df.iloc[X_test.index].copy()
    
    return X_train_scaled, X_test_scaled, y_train, y_test, test_data_original

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_clustering(filepath):
    df = pd.read_csv(filepath)
    
    # Clustering ไม่ต้องใช้คำตอบ (Target) เราจะเลือกมาแค่คอลัมน์ที่เป็นตัวเลขเท่านั้น
    X = df.select_dtypes(include=['number'])
    
    # ทำ Standardization ปรับสเกลข้อมูล
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # คืนค่าข้อมูลต้นฉบับกลับไปด้วยเพื่อใช้ออกรายงาน
    return X_scaled, df.copy(), X.columns
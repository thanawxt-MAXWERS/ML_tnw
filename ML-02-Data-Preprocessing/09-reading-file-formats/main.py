import pandas as pd
import os

def read_formats():
    print("=== Reading & Writing File Formats ===")
    dri_path = os.path.join(os.path.dirname(__file__), '..', 'drivers.csv')
    df = pd.read_csv(dri_path)
    
    json_path = 'drivers_temp.json'
    df.head().to_json(json_path, orient='records')
    print(f"บันทึกไฟล์ 5 แถวแรกเป็น JSON -> {json_path}")
    
    df_json = pd.read_json(json_path)
    print("\nข้อมูลที่อ่านมาจาก JSON:")
    print(df_json[['driverId', 'driverRef']])
    
    if os.path.exists(json_path):
        os.remove(json_path)

if __name__ == "__main__":
    read_formats()
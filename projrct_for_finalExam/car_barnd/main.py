import os
from src.camera import run_realtime_detection

# แก้ไขชื่อแบรนด์ให้ตรงกับที่ Print ออกมาตอนรันเทรนโมเดล
CLASS_NAMES = ['Acura', 'Aston Martin', 'Audi', 'Bentley', 'BMW', 'Buick', 
    'Cadillac', 'Chevrolet', 'Chrysler', 'Dodge', 'FIAT', 'Ford', 
    'GMC', 'Honda', 'Hyundai', 'INFINITI', 'Jaguar', 'Jeep', 'Kia', 
    'Land Rover', 'Lexus', 'Lincoln', 'Mazda', 'Mercedes-Benz', 
    'MINI', 'Mitsubishi', 'Nissan', 'Porsche', 'Ram', 'Subaru', 
    'Toyota', 'Volkswagen', 'Volvo']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'car_brand_model.keras')

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] ไม่พบไฟล์โมเดลที่ '{MODEL_PATH}'")
        print("กรุณารันไฟล์ 'python src/train.py' เพื่อเทรนสร้างโมเดลก่อน")
    else:
        run_realtime_detection(MODEL_PATH, CLASS_NAMES, threshold=0.40)
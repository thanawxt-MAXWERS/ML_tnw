import os
import cv2
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'car_brand_model.keras')

CLASS_NAMES = [
    'Acura', 'Aston Martin', 'Audi', 'Bentley', 'BMW', 'Buick', 
    'Cadillac', 'Chevrolet', 'Chrysler', 'Dodge', 'FIAT', 'Ford', 
    'GMC', 'Honda', 'Hyundai', 'INFINITI', 'Jaguar', 'Jeep', 'Kia', 
    'Land Rover', 'Lexus', 'Lincoln', 'Mazda', 'Mercedes-Benz', 
    'MINI', 'Mitsubishi', 'Nissan', 'Porsche', 'Ram', 'Subaru', 
    'Toyota', 'Volkswagen', 'Volvo'
]

def predict_and_save_batch(input_folder, output_folder):
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] ไม่พบไฟล์โมเดลที่ '{MODEL_PATH}'")
        return

    if not os.path.exists(input_folder):
        print(f"[ERROR] ไม่พบโฟลเดอร์รูปภาพต้นทางที่ '{input_folder}'")
        print("กรุณาสร้างโฟลเดอร์ 'test_images' แล้วเอารูปไปวางไว้ก่อนครับ")
        return

    # สร้างโฟลเดอร์ปลายทางให้อัตโนมัติหากยังไม่มี
    os.makedirs(output_folder, exist_ok=True)

    print("[INFO] กำลังโหลดโมเดล...")
    model = tf.keras.models.load_model(MODEL_PATH)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"[WARN] ไม่พบไฟล์รูปภาพในโฟลเดอร์: {input_folder}")
        return

    print(f"\n================ กำลังประมวลผลรูปภาพทั้งหมด ({len(image_files)} รูป) ================\n")

    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"[ERROR] อ่านไฟล์ {filename} ไม่ได้")
            continue

        # แปลงสีและปรับขนาดภาพเข้าโมเดล
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (224, 224))
        img_array = np.expand_dims(resized_frame, axis=0)

        # ทำนายผล
        predictions = model.predict(img_array, verbose=0)
        class_index = np.argmax(predictions[0])
        confidence = predictions[0][class_index]
        predicted_brand = CLASS_NAMES[class_index]

        # คำนวณขนาดตัวอักษรให้พอดีกับขนาดรูปภาพอัตโนมัติ
        h, w = frame.shape[:2]
        font_scale = max(0.6, min(w, h) / 700)
        thickness = max(1, int(font_scale * 2))

        # วาดแถบพื้นหลังสีดำและเขียนข้อความแบรนด์
        display_text = f"{predicted_brand}: {confidence * 100:.1f}%"
        bg_width = int(w * 0.65)
        bg_height = int(h * 0.1)
        
        cv2.rectangle(frame, (10, 10), (bg_width, bg_height), (0, 0, 0), -1)
        cv2.putText(frame, display_text, (20, int(bg_height * 0.7)), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

        # บันทึกภาพลงโฟลเดอร์ output_images
        save_path = os.path.join(output_folder, f"result_{filename}")
        cv2.imwrite(save_path, frame)

        print(f"✅ บันทึกสำเร็จ: {filename} -> result_{filename} ({predicted_brand}: {confidence * 100:.1f}%)")

    print(f"\n[INFO] เสร็จสิ้น! ไฟล์รูปภาพทั้งหมดถูกเซฟไว้ที่: {output_folder}\n")

if __name__ == "__main__":
    INPUT_FOLDER = os.path.join(BASE_DIR, 'test_images')
    OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output_images')
    predict_and_save_batch(INPUT_FOLDER, OUTPUT_FOLDER)
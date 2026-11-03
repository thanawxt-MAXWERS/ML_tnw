import cv2
import numpy as np
import tensorflow as tf

def run_realtime_detection(model_path, class_names, threshold=0.70):
    print(f"[INFO] กำลังโหลดโมเดลจาก: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"[ERROR] โหลดโมเดลไม่ได้: {e}")
        return

    cap = cv2.VideoCapture(0)
    
    # 1. ลดความละเอียดกล้องลงให้ต่ำที่สุดที่ยังพอมองเห็น (ช่วยลดภาระ CPU ได้มาก)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():
        print("[ERROR] ไม่สามารถเชื่อมต่อกล้องได้")
        return

    print("[INFO] เปิดกล้องสำเร็จ (กด 'q' เพื่อออก)")
    
    frame_count = 0
    current_brand = ""
    current_confidence = 0.0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1

        # 2. ปรับให้ประมวลผลแค่ 1 ครั้ง ในทุกๆ 15 เฟรม
        if frame_count % 15 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(rgb_frame, (224, 224))
            img_array = np.expand_dims(resized_frame, axis=0)

            predictions = model.predict(img_array, verbose=0)
            class_index = np.argmax(predictions[0])
            current_confidence = predictions[0][class_index]
            
            if current_confidence >= threshold:
                current_brand = class_names[class_index]
            else:
                current_brand = ""

        if current_brand != "":
            display_text = f"{current_brand}: {current_confidence * 100:.1f}%"
            cv2.rectangle(frame, (10, 10), (300, 50), (0, 0, 0), -1)
            # ลดขนาดตัวอักษรลงนิดหน่อยให้เหมาะกับจอเล็ก
            cv2.putText(frame, display_text, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ขยายภาพกลับให้ดูบนจอได้ถนัดขึ้น (ไม่มีผลกับ CPU ตอนทำนาย)
        display_frame = cv2.resize(frame, (640, 480))
        cv2.imshow("Car Brand AI Detector", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
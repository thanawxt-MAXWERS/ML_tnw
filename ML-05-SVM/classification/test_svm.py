import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"
IMG_SIZE = 100  # ขนาดรูปภาพที่ใช้เทรน (ตรงกับใน main.py)

def main():
    print("--------------------------------------------------")
    print("       Testing Model & Generating Samples         ")
    print("--------------------------------------------------\n")
    
    # 1. โหลดโมเดล, Scaler และข้อมูลทดสอบที่บันทึกไว้
    model = joblib.load(f"{OUTPUT_DIR}/svm_model.pkl")
    scaler = joblib.load(f"{OUTPUT_DIR}/scaler.pkl")
    X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")
    
    with open(f"{OUTPUT_DIR}/classes.json", "r") as f:
        classes = json.load(f)
        
    # 2. ทำนายผลข้อมูลทดสอบ
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    
    # 3. สุ่มตัวอย่างรูปภาพมาแสดงผล 4 รูป และบันทึกเป็นไฟล์ภาพ
    print("Generating prediction samples image...")
    plt.figure(figsize=(8, 8))
    sample_indices = np.random.choice(len(X_test), min(4, len(X_test)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        plt.subplot(2, 2, i + 1)
        # แปลงข้อมูลฟีเจอร์กลับเป็นรูปภาพ 2D (IMG_SIZE x IMG_SIZE)
        img_view = X_test[idx].reshape(IMG_SIZE, IMG_SIZE)
        plt.imshow(img_view, cmap='gray')
        
        true_label = classes[y_test[idx]]
        pred_label = classes[predictions[idx]]
        
        # กำหนดสีข้อความ: เขียวถ้าทายถูก, แดงถ้าทายผิด
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=12)
        plt.axis('off')
        
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'prediction_samples.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Successfully saved: {output_path}")

if __name__ == "__main__":
    main()
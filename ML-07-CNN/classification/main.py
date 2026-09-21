import json
import os
import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from cnn_model import train_model, predict_model
from evaluate import evaluate_model, plot_history

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "dataimmage")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# --- จุดแก้ Hyperparameter ชุดที่ 1 ---
IMG_SIZE = 128
TEST_SIZE = 0.2
VAL_SIZE = 0.2       # เปลี่ยนจาก 0.1 เป็น 0.2 (เพิ่มชุดข้อสอบให้ใหญ่ขึ้น กราฟจะนิ่งขึ้น)
MAX_PER_CLASS = 3000
EPOCHS = 32          # 32 หรือ 40 รอบก็กำลังดีครับ
BATCH_SIZE = 32      # เปลี่ยนจาก 16 เป็น 32 (จุดสมดุลที่ดีที่สุด ไม่แกว่งไปและไม่อืดไป)
# --------------------------------------

def main():
    print("--" * 30)
    print("CNN Image Recognition: Cars vs Motorcycles")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)
    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\n[Step 2] Preprocessing images...")
    X = to_features(images)
    y = labels
    np.save(f"{OUTPUT_DIR}/features.npy", X)

    print("\n[Step 3] Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, TEST_SIZE, VAL_SIZE)
    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_val.npy", X_val)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_val.npy", y_val)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print("\n[Step 4] Training model...")
    model, history = train_model(X_train, y_train, X_val, y_val, len(classes), OUTPUT_DIR, EPOCHS, BATCH_SIZE)

    print("\n[Step 5] Testing model...")
    predictions = predict_model(model, X_test)

    print("\n[Step 6] Evaluating model...")
    evaluate_model(y_test, predictions, classes, save_path=f"{OUTPUT_DIR}/confusion_matrix.png")
    plot_history(history, f"{OUTPUT_DIR}/training_history.png")

if __name__ == "__main__":
    main()
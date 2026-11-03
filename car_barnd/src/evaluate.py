import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(BASE_DIR, 'dataset', 'test')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'car_brand_model.keras')
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def evaluate_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] ไม่พบโมเดลที่: {MODEL_PATH}")
        return

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, shuffle=False, batch_size=BATCH_SIZE, image_size=IMG_SIZE
    )
    class_names = test_ds.class_names
    model = tf.keras.models.load_model(MODEL_PATH)
    
    y_true, y_pred = [], []
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        pred_labels = np.argmax(predictions, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(pred_labels)

    print("\n================ Classification Report ================")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Car Brand AI')
    plt.ylabel('Actual Brand')
    plt.xlabel('Predicted Brand')
    
    save_path = os.path.join(BASE_DIR, 'confusion_matrix.png')
    plt.savefig(save_path)
    print(f"[INFO] บันทึกรูป Confusion Matrix ที่: {save_path}")
    plt.show()

if __name__ == "__main__":
    evaluate_model()
import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'train')
VAL_DIR = os.path.join(BASE_DIR, 'dataset', 'val')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'car_brand_model.keras')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

def train_model():
    print("[INFO] กำลังโหลดชุดข้อมูลรูปภาพ...")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
    )

    class_names = train_ds.class_names
    print(f"[INFO] แบรนด์ที่พบ: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("[INFO] เริ่มเทรนโมเดล...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"[INFO] บันทึกโมเดลสำเร็จที่: {MODEL_SAVE_PATH}")
    print(f"[สำคัญ] นำรายชื่อนี้ไปใส่ใน main.py: {class_names}")

if __name__ == "__main__":
    train_model()
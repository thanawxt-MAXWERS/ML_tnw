import os
import cv2
import numpy as np
from preprocessing import preprocess_image

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

def load_data(data_path, img_size=100, max_per_class=None):
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {os.path.abspath(data_path)}\n"
            "Expected dataimmage/Cars/ and dataimmage/Motorcycles/ at the project root."
            #"Expected dataimmage/Bikes/ and dataimmage/Motorcycles/ at the project root."
        )

    images = []
    labels = []
    classes = sorted([folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))])

    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_path, class_name)
        filenames = sorted(f for f in os.listdir(class_path) if f.lower().endswith(VALID_EXT))
        loaded, skipped = 0, 0
        
        for filename in filenames:
            if max_per_class and loaded >= max_per_class: break
            image = cv2.imread(os.path.join(class_path, filename))
            image = preprocess_image(image, img_size)
            if image is None:
                skipped += 1
                continue
            images.append(image)
            labels.append(label)
            loaded += 1
            
    return np.stack(images), np.array(labels), classes
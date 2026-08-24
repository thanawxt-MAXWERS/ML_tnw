import os
import cv2
import numpy as np

def get_valid_image_paths(data_dir):
    paths = []
    labels = []
    classes = ['Pistols', 'Rifles', 'shotguns']
    
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path):
            continue
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                paths.append(img_path)
                labels.append(class_idx)
                
    return paths, np.array(labels), classes
import cv2
import numpy as np
import json
import os

def preprocess_and_save(paths, labels, classes, output_dir, img_size=(64, 64)):
    features = []
    
    for path in paths:
        img = cv2.imread(path)
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        features.append(img)
        
    features = np.array(features, dtype='float32') / 255.0
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'features.npy'), features)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)
    
    with open(os.path.join(output_dir, 'classes.json'), 'w') as f:
        json.dump(classes, f)
        
    return features, labels
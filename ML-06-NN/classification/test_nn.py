import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tensorflow.keras.models import load_model

def test_random_images(output_dir):
    X_test = np.load(os.path.join(output_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(output_dir, 'y_test.npy'))
    
    with open(os.path.join(output_dir, 'classes.json'), 'r') as f:
        classes = json.load(f)
        
    model = load_model(os.path.join(output_dir, 'nn_model_2.keras'))
    
    indices = np.random.choice(len(X_test), min(4, len(X_test)), replace=False)
    
    plt.figure(figsize=(10, 8))
    for i, idx in enumerate(indices):
        img = X_test[idx]
        true_label = classes[y_test[idx]]
        
        pred_prob = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
        pred_idx = np.argmax(pred_prob)
        pred_label = classes[pred_idx]
        
        plt.subplot(2, 2, i+1)
        plt.imshow(img)
        plt.title(f"True: {true_label} | Pred: {pred_label}\nProb: {pred_prob[pred_idx]:.4f}")
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_sample.png'))
    plt.show()
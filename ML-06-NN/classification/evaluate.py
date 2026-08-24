import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
from tensorflow.keras.models import load_model

def evaluate_model(output_dir):
    X_test = np.load(os.path.join(output_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(output_dir, 'y_test.npy'))
    
    with open(os.path.join(output_dir, 'classes.json'), 'r') as f:
        classes = json.load(f)
        
    model_1 = load_model(os.path.join(output_dir, 'nn_model_1.keras'))
    model_2 = load_model(os.path.join(output_dir, 'nn_model_2.keras'))
    
    loss_1, acc_1 = model_1.evaluate(X_test, y_test, verbose=0)
    loss_2, acc_2 = model_2.evaluate(X_test, y_test, verbose=0)
    
    print("\n[Accuracy Scores for each NN Configuration]")
    print(f"Model 1 (1 Hidden Layer, 15 Epochs)  : {acc_1:.4f}")
    print(f"Model 2 (2 Hidden Layers, 30 Epochs) : {acc_2:.4f}\n")
    
    y_pred_prob_2 = model_2.predict(X_test, verbose=0)
    y_pred_2 = np.argmax(y_pred_prob_2, axis=1)
    
    cm = confusion_matrix(y_test, y_pred_2)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Model 2)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    with open(os.path.join(output_dir, 'history_1.json'), 'r') as f:
        h1 = json.load(f)
    with open(os.path.join(output_dir, 'history_2.json'), 'r') as f:
        h2 = json.load(f)
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(h1['loss'], label='Train Loss')
    axes[0, 0].plot(h1['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model 1 Loss (15 Epochs)')
    axes[0, 0].legend()
    
    axes[0, 1].plot(h1['accuracy'], label='Train Acc')
    axes[0, 1].plot(h1['val_accuracy'], label='Val Acc')
    axes[0, 1].set_title('Model 1 Accuracy (15 Epochs)')
    axes[0, 1].legend()
    
    axes[1, 0].plot(h2['loss'], label='Train Loss')
    axes[1, 0].plot(h2['val_loss'], label='Val Loss')
    axes[1, 0].set_title('Model 2 Loss (30 Epochs)')
    axes[1, 0].legend()
    
    axes[1, 1].plot(h2['accuracy'], label='Train Acc')
    axes[1, 1].plot(h2['val_accuracy'], label='Val Acc')
    axes[1, 1].set_title('Model 2 Accuracy (30 Epochs)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    
    print("[Classification Report - Model 2]")
    print(classification_report(y_test, y_pred_2, target_names=classes))
import numpy as np
from sklearn.model_selection import train_test_split
import os

def split_and_save(output_dir):
    X = np.load(os.path.join(output_dir, 'features.npy'))
    y = np.load(os.path.join(output_dir, 'labels.npy'))
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
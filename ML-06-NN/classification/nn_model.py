import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
import json
import os

def build_model_1(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model_2(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_save(X_train, y_train, X_val, y_val, output_dir):
    model_1 = build_model_1(X_train.shape[1:])
    history_1 = model_1.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val), verbose=1)
    model_1.save(os.path.join(output_dir, 'nn_model_1.keras'))
    
    model_2 = build_model_2(X_train.shape[1:])
    history_2 = model_2.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), verbose=1)
    model_2.save(os.path.join(output_dir, 'nn_model_2.keras'))
    
    with open(os.path.join(output_dir, 'history_1.json'), 'w') as f:
        json.dump(history_1.history, f)
        
    with open(os.path.join(output_dir, 'history_2.json'), 'w') as f:
        json.dump(history_2.history, f)
        
    return model_1, model_2
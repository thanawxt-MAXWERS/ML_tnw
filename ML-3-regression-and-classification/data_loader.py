import os
import pandas as pd
import numpy as np

def prepare_data_and_meta(csv_path='age_gender.csv', output_dir='others_dir'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(csv_path)
    
    # สร้าง meta.csv
    meta_df = df.drop(columns=['pixels']).reset_index()
    meta_df.rename(columns={'index': 'id'}, inplace=True)
    meta_df.to_csv(os.path.join(output_dir, 'meta.csv'), index=False)
    
    # สร้าง pixels.npy
    pixels = np.array([np.array(x.split(), dtype='float32') for x in df['pixels']])
    np.save(os.path.join(output_dir, 'pixels.npy'), pixels)
    
    return pixels, df['age'].values, df['gender'].values

def load_data(output_dir='others_dir'):
    pixels_path = os.path.join(output_dir, 'pixels.npy')
    meta_path = os.path.join(output_dir, 'meta.csv')
    
    if os.path.exists(pixels_path) and os.path.exists(meta_path):
        X = np.load(pixels_path)
        meta_df = pd.read_csv(meta_path)
        y_age = meta_df['age'].values
        y_gender = meta_df['gender'].values
        return X, y_age, y_gender
    else:
        return prepare_data_and_meta()
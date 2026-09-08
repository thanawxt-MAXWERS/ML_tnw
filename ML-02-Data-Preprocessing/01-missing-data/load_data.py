import pandas as pd
import numpy as np
import os

def load_f1_data():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    res_path = os.path.join(base_dir, 'results.csv')
    dri_path = os.path.join(base_dir, 'drivers.csv')
    
    df_res = pd.read_csv(res_path)
    df_dri = pd.read_csv(dri_path)
    df = pd.merge(df_res, df_dri, on='driverId', how='left')
    
    # แปลง \N เป็นค่าว่าง (NaN)
    df.replace(r'\\N', np.nan, regex=True, inplace=True)
    return df
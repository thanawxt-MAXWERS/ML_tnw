import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate(y_true, y_pred, output_dir='regression/outputs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print(f"R2 Score: {r2_score(y_true, y_pred):.2f}")
    
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.title('Age Prediction Results')
    plt.savefig(os.path.join(output_dir, 'regression_results.png'))
    plt.close()
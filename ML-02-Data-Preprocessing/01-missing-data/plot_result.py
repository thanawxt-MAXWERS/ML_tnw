import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_imputation(original, imputed, title):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(original, kde=True, color='red', bins=30)
    plt.title('Original (with Missing)')
    
    plt.subplot(1, 2, 2)
    sns.histplot(imputed, kde=True, color='green', bins=30)
    plt.title(title)
    
    plt.tight_layout()
    
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{title.replace(" ", "_")}.png'))
    plt.show()
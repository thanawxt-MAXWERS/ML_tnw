from load_data import load_f1_data
from plot_result import plot_imputation

def handle_missing_data():
    print("=== Part 3: Missing Value Handling ===")
    df = load_f1_data()
    
    # ใช้คอลัมน์ number_x (หมายเลขรถ) เป็นตัวทดสอบ
    col = 'number_x'
    if col in df.columns:
        df[col] = df[col].astype(float) # แปลงเป็นตัวเลขก่อน
        print(f"Missing values before: {df[col].isnull().sum()}")
        
        original = df[col].dropna()
        df_mean = df.copy()
        df_median = df.copy()
        
        mean_val = df[col].mean()
        median_val = df[col].median()
        
        df_mean[col].fillna(mean_val, inplace=True)
        df_median[col].fillna(median_val, inplace=True)
        
        print(f"Filled with Mean: {mean_val:.2f}")
        print(f"Filled with Median: {median_val:.2f}")
        
        plot_imputation(original, df_mean[col], 'Mean Imputation')
        plot_imputation(original, df_median[col], 'Median Imputation')

if __name__ == "__main__":
    handle_missing_data()
from data_loader import load_data
import regression.main as reg_main
import classification.main as clf_main

if __name__ == "__main__":
    print("Loading data...")
    X, y_age, y_gender = load_data()
    print(f"Data loaded: {X.shape[0]} images")
    
    print("\n--- Starting Regression (Age Prediction) ---")
    reg_main.run(X, y_age)
    
    print("\n--- Starting Classification (Gender Prediction) ---")
    clf_main.run(X, y_gender)
    
    print("\nAll tasks completed successfully!")
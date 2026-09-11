from sklearn.model_selection import train_test_split
from .model import build_model
from .evaluate import evaluate

def run(X, y_gender):
    X_train, X_test, y_train, y_test = train_test_split(X, y_gender, test_size=0.2, random_state=42)
    
    model = build_model(n_components=100)
    print("Training Classification model...")
    model.fit(X_train, y_train)
    
    print("Evaluating Classification model...")
    y_pred = model.predict(X_test)
    evaluate(y_test, y_pred)
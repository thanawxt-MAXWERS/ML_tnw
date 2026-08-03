import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import os
from knn_tf import train_knn_model, predict_knn

def evaluate_and_save_outputs(X_train, X_test, y_train, y_test, test_data_original, k_values):
    os.makedirs('outputs', exist_ok=True)
    
    accuracies = []
    best_k = k_values[0]
    best_acc = 0
    best_model = None
    best_preds = None

    for k in k_values:
        model = train_knn_model(X_train, y_train, k)
        preds = predict_knn(model, X_test)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)
        
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_model = model
            best_preds = preds

    print(f"ค่า k ที่ดีที่สุดคือ {best_k} (Accuracy = {best_acc:.4f})")

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracies, marker='o', linestyle='dashed', color='b')
    plt.title('Accuracy for Different K Values')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.grid(True)
    plt.savefig('outputs/01_k_curve.png')
    plt.close()

    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Best k={best_k})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('outputs/02_confusion_matrix.png')
    plt.close()

    test_data_original['Actual'] = y_test.values
    test_data_original['Predicted'] = best_preds
    test_data_original.to_csv('outputs/predictions.csv', index=False)
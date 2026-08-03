from sklearn.neighbors import KNeighborsClassifier

def train_knn_model(X_train, y_train, k_value):
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(X_train, y_train)
    return knn

def predict_knn(model, X_test):
    return model.predict(X_test)
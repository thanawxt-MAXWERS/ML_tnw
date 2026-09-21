import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

def test_cnn(n_samples=4):
    model = keras.models.load_model(f"{OUTPUT_DIR}/cnn_model.keras")
    X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")
    with open(f"{OUTPUT_DIR}/classes.json") as f:
        classes = json.load(f)

    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample, y_sample = X_test[index], y_test[index]
    probabilities = model.predict(X_sample, verbose=0)
    
    if probabilities.shape[-1] == 1:
        probabilities = probabilities.ravel()
        predictions = (probabilities > 0.5).astype(int)
        confidence = np.where(predictions == 1, probabilities, 1 - probabilities)
    else:
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)

    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 4.0 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i >= n_samples:
            ax.axis("off")
            continue
        pred, true = classes[predictions[i]], classes[y_sample[i]]
        correct = predictions[i] == y_sample[i]
        
        ax.imshow(X_sample[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Pred: {pred} ({confidence[i] * 100:.0f}%)\nTrue: {true}", color="green" if correct else "red")

    correct_total = int((predictions == y_sample).sum())
    fig.suptitle(f"Prediction: {correct_total}/{n_samples} correct")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/prediction_sample.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    test_cnn()
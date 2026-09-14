## Structure

```text
ML-05-SVM/
│
├── classification/
│   ├── Rock Paper Scissors/
│   │   └── Rock Paper Scissors/
│   │       ├── paper/
│   │       │   ├── 0.jpg 
│   │       │   └── ...
│   │       ├── rock/
│   │       │   ├── 0.jpg
│   │       │   └── ...
│   │       └── scissors/
│   │           ├── 0.jpg
│   │           └── ...
│   │
│   ├── main.py
│   ├── test_svm.py
│   ├── data_load.py
│   ├── preprocess.py
│   ├── split_data.py
│   ├── svm_model.py
│   ├── evaluate.py
│   ├── draw_flow.py
│   └── outputs/
│       ├── images.npy
│       ├── labels.npy
│       ├── classes.json
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       ├── scaler.pkl
│       ├── svm_model.pkl
│       ├── confusion_matrix.png
│       └── prediction_samples.png
│
├── requirements.txt
└── link-data.txt

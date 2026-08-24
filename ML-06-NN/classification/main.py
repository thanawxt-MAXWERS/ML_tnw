import os
from data_loader import get_valid_image_paths
from preprocessing import preprocess_and_save
from split_data import split_and_save
from nn_model import train_and_save
from evaluate import evaluate_model
from test_nn import test_random_images

def main():
    DATA_DIR = '../wimage' 
    OUTPUT_DIR = 'outputs'
    
    paths, labels, classes = get_valid_image_paths(DATA_DIR)
    preprocess_and_save(paths, labels, classes, OUTPUT_DIR, img_size=(64, 64))
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_save(OUTPUT_DIR)
    train_and_save(X_train, y_train, X_val, y_val, OUTPUT_DIR)
    evaluate_model(OUTPUT_DIR)
    test_random_images(OUTPUT_DIR)

if __name__ == "__main__":
    main()
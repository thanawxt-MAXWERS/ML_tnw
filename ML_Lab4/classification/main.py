import os
from data_loader import load_and_preprocess
from evaluate import evaluate_and_save_outputs

def main():
    # 1. ล็อคตำแหน่งไฟล์แบบตายตัว (Absolute Path)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = r"D:\work\myW\ML_Lab4\data-F1\F1Drivers_Dataset.csv"
    target_col = 'Champion'
    
    k_values = [3, 5, 7, 9, 11]

    print("กำลังโหลดและประมวลผลข้อมูล...")
    X_train, X_test, y_train, y_test, test_data_original = load_and_preprocess(filepath, target_col)

    print("กำลังประเมินผลและสร้างไฟล์ Output...")
    evaluate_and_save_outputs(X_train, X_test, y_train, y_test, test_data_original, k_values)
    
    print("เสร็จสิ้น! ตรวจสอบไฟล์ผลลัพธ์ได้ที่โฟลเดอร์ classification/outputs/")

if __name__ == '__main__':
    main()

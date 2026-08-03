import os
from data_loader import load_and_preprocess_clustering
from kmeans_tf import calculate_wcss, train_kmeans
from visualize import save_clustering_outputs

def main():
    # กำหนด Path ไปหาไฟล์ F1 (ล็อคตำแหน่งตายตัว)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, '..', 'data-F1', 'F1Drivers_Dataset.csv')
    filepath = os.path.normpath(filepath)

    print("กำลังโหลดและประมวลผลข้อมูล (Clustering)...")
    X_scaled, original_data, feature_names = load_and_preprocess_clustering(filepath)

    print("กำลังคำนวณ Elbow Method เพื่อหาจำนวนกลุ่ม (K) ที่เหมาะสม...")
    k_values, wcss = calculate_wcss(X_scaled, max_k=10)
    
    # เราเลือก k=3 สำหรับการจัดกลุ่ม (ปกติวิเคราะห์จากจุดหักศอกของกราฟ Elbow)
    optimal_k = 3
    print(f"ทำการจัดกลุ่มข้อมูลเป็น {optimal_k} กลุ่มด้วย K-Means...")
    kmeans_model, clusters = train_kmeans(X_scaled, n_clusters=optimal_k)

    print("กำลังสร้างกราฟและบันทึกผลลัพธ์ลงโฟลเดอร์ outputs...")
    save_clustering_outputs(k_values, wcss, original_data, clusters, X_scaled, feature_names)

    print("✅ เสร็จสิ้น! ตรวจสอบไฟล์ผลลัพธ์ได้ที่โฟลเดอร์ clustering/outputs/")

if __name__ == '__main__':
    main()
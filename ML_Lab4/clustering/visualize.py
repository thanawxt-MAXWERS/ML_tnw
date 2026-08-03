import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_clustering_outputs(k_values, wcss, original_data, clusters, X_scaled, feature_names):
    os.makedirs('outputs', exist_ok=True)
    
    # 1. Elbow Curve (01_elbow.png)
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, wcss, marker='o', linestyle='dashed', color='red')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS')
    plt.xticks(k_values)
    plt.grid(True)
    plt.savefig('outputs/01_elbow.png')
    plt.close()
    
    # 2. Scatter Plot ของ Clusters (02_clusters.png)
    # พล็อต 2 ฟีเจอร์แรกเพื่อให้เห็นการกระจายตัว
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters, palette='Set2', s=100)
    plt.title('K-Means Clusters Distribution')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend(title='Cluster')
    plt.savefig('outputs/02_clusters.png')
    plt.close()
    
    # 3. ไฟล์ข้อมูลที่ถูกจัดกลุ่มแล้ว (clustered_f1_drivers.csv)
    original_data['Cluster'] = clusters
    original_data.to_csv('outputs/clustered_f1_drivers.csv', index=False)
    
    # 4. สรุปค่าเฉลี่ยของแต่ละกลุ่ม (cluster_summary.csv)
    summary = original_data.select_dtypes(include=['number']).groupby('Cluster').mean()
    summary.to_csv('outputs/cluster_summary.csv')
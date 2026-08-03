from sklearn.cluster import KMeans

def calculate_wcss(X_scaled, max_k=10):
    # หาค่า WCSS สำหรับพล็อต Elbow Curve
    wcss = []
    k_values = range(1, max_k + 1)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    return k_values, wcss

def train_kmeans(X_scaled, n_clusters):
    # เทรนและจัดกลุ่มข้อมูลจริง
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    return kmeans, clusters
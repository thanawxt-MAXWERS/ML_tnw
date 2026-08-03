from sklearn.neighbors import NearestNeighbors

def find_nearest_neighbors(X_scaled, n_neighbors=3):
    # ใช้หลักการ KNN หาจุดที่อยู่ใกล้เคียงกันที่สุด (คล้ายการหาระยะห่าง)
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)
    return distances, indices
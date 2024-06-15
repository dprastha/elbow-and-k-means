import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Data Contoh (dataset hipotetis)
data = pd.DataFrame({
    'Age': [25, 34, 22, 27, 45, 52, 23, 38, 46, 55],
    'Annual Income (k$)': [15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
    'Spending Score (1-100)': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72]
})

# Standarisasi fitur
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Gunakan Metode Elbow untuk Menentukan k Optimal
inertia = []
K = range(1, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

# Plotting the elbow curve
plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

# Setelah melihat plot elbow, kita pilih k optimal (misalnya k=3)
optimal_k = 3

# Melakukan K-means dengan k Optimal
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(data_scaled)
clusters = kmeans.predict(data_scaled)
data['Cluster'] = clusters

# Visualisasi hasil clustering
sns.pairplot(data, hue='Cluster', palette='viridis')
plt.show()

# Tampilkan data dengan cluster masing-masing
print(data)

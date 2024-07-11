import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
SONGS_FILE = "src/spotify.csv"
df = pd.read_csv(SONGS_FILE)

# Extract relevant features
features = df[['popularity', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using the elbow method
# wcss = []  # Within-cluster sum of squares
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, random_state=42)
#     kmeans.fit(scaled_features)
#     wcss.append(kmeans.inertia_)

# # Plot the elbow method
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, 11), wcss, marker='o')
# plt.title('Elbow Method for Optimal Number of Clusters')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()



kmeans = KMeans(n_clusters=10, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)

# Print 5 songs and their artists from each cluster
for cluster in range(10):
    print(f"Cluster {cluster}:")
    cluster_data = df[df['cluster'] == cluster]
    for idx, row in cluster_data.head(5).iterrows():
        print(f"  Song: {row['track_name']}, Artist: {row['artists']}")
    print()

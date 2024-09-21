import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fcmeans import FCM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    X = data[['Age', 'Spending Score (1-100)', 'Annual Income (k$)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X


def determine_optimal_clusters(X_scaled, max_k=10):
    wcss = []
    silhouette_scores = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))


    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_k + 1), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')


    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Scores')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')

    plt.show()

    return silhouette_scores.index(max(silhouette_scores)) + 2


def run_clustering(strategy, X_scaled):
    labels, centers, membership = strategy.cluster(X_scaled)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=labels, cmap='viridis', s=50, alpha=0.6, edgecolors='w')
    
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', marker='x', s=100, label='Centroids')


    ax.set_title('Fuzzy K-Means Clustering (3D View)')
    ax.set_xlabel('Age')
    ax.set_ylabel('Spending Score')
    ax.set_zlabel('Annual Income')

    plt.legend()
    plt.colorbar(scatter, label='Cluster Labels')
    plt.show()

    return labels, membership

def evaluate_clustering(X_scaled, labels, membership):

    sil_score = silhouette_score(X_scaled, labels)
    print(f'Silhouette Score: {sil_score:.2f}')

    print("\nMembership degrees (top 5 samples):")
    print(membership[:5])

class FuzzyCMeansStrategy:
    def __init__(self, n_clusters):
        self.fcm = FCM(n_clusters=n_clusters)

    def cluster(self, X):
        self.fcm.fit(X)
        labels = self.fcm.u.argmax(axis=1)
        centers = self.fcm.centers
        return labels, centers, self.fcm.u

def main():
    X_scaled, X = preprocess_data('Mall_customers.csv')

    optimal_k = determine_optimal_clusters(X_scaled, max_k=10)
    print(f'Optimal number of clusters determined: {optimal_k}')

    clustering_strategy = FuzzyCMeansStrategy(n_clusters=optimal_k)

    labels, membership = run_clustering(clustering_strategy, X_scaled)

    evaluate_clustering(X_scaled, labels, membership)

if __name__ == "__main__":
    main()

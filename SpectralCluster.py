import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

class ClusteringStrategy:
    def cluster(self, X):
        raise NotImplementedError

class SpectralClusteringStrategy(ClusteringStrategy):
    def __init__(self, n_clusters=3):
        self.spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=0)

    def cluster(self, X):
        labels = self.spectral.fit_predict(X)
        return labels

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    X = data[['Annual Income (k$)', 'Spending Score (1-100)']]  
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X

def run_clustering(strategy, X_scaled, X):
    labels = strategy.cluster(X_scaled)

    plt.figure(figsize=(10, 7))

    scatter = plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=labels, cmap='viridis', s=50, alpha=0.6, edgecolors='w')

    plt.title('Spectral Clustering (2D View)')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score')

    plt.colorbar(scatter, label='Cluster Labels')
    plt.show()

    return labels

# Evaluation Function: Silhouette Score
def evaluate_clustering(X_scaled, labels):
    sil_score = silhouette_score(X_scaled, labels)
    print(f'Silhouette Score: {sil_score:.2f}')

def main():
    X_scaled, X = preprocess_data('Mall_Customers.csv')

    clustering_strategy = SpectralClusteringStrategy(n_clusters=5)

    labels = run_clustering(clustering_strategy, X_scaled, X)

    evaluate_clustering(X_scaled, labels)

if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

# Clustering Strategy Interface
class ClusteringStrategy:
    def cluster(self, X):
        raise NotImplementedError

# Spectral Clustering Strategy
class SpectralClusteringStrategy(ClusteringStrategy):
    def __init__(self, n_clusters=3):
        self.spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=0)

    def cluster(self, X):
        labels = self.spectral.fit_predict(X)
        return labels

# Data Preprocessing Function
def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    X = data[['Age', 'Spending Score (1-100)']]  # Only use Age and Spending Score
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X

# 2D Clustering and Visualization Function
def run_clustering(strategy, X_scaled, X):
    labels = strategy.cluster(X_scaled)

    # Create 2D scatter plot
    plt.figure(figsize=(10, 7))

    # Plot 2D scatter: Age vs Spending Score
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6, edgecolors='w')

    # Set axis labels and title
    plt.title('Spectral Clustering (2D View)')
    plt.xlabel('Age')
    plt.ylabel('Spending Score')

    # Add color bar
    plt.colorbar(scatter, label='Cluster Labels')
    plt.show()

    return labels

# Evaluation Function: Silhouette Score
def evaluate_clustering(X_scaled, labels):
    # Calculate silhouette score
    sil_score = silhouette_score(X_scaled, labels)
    print(f'Silhouette Score: {sil_score:.2f}')

# Main function to run the spectral clustering and evaluation
def main():
    # Preprocess the data
    X_scaled, X = preprocess_data('Mall_customers.csv')

    # Choose the clustering strategy (Spectral Clustering here)
    clustering_strategy = SpectralClusteringStrategy(n_clusters=5)

    # Run the clustering and 2D visualization
    labels = run_clustering(clustering_strategy, X_scaled, X)

    # Evaluate the clustering result
    evaluate_clustering(X_scaled, labels)

# Execute the main function
if __name__ == "__main__":
    main()

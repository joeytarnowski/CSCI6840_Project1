import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fcmeans import FCM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Clustering Strategy Interface
class ClusteringStrategy:
    def cluster(self, X):
        raise NotImplementedError

# Fuzzy C-Means Strategy
class FuzzyCMeansStrategy(ClusteringStrategy):
    def __init__(self, n_clusters=3):
        self.fcm = FCM(n_clusters=n_clusters)

    def cluster(self, X):
        self.fcm.fit(X)
        labels = self.fcm.u.argmax(axis=1)
        centers = self.fcm.centers
        return labels, centers, self.fcm.u

# Data Preprocessing Function
def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    X = data[['Age', 'Spending Score (1-100)']]  # Only use Age and Spending Score
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X

# 2D Clustering and Visualization Function
def run_clustering(strategy, X_scaled, X):
    labels, centers, membership = strategy.cluster(X_scaled)

    # Create 2D scatter plot
    plt.figure(figsize=(10, 7))

    # Plot 2D scatter: Age vs Spending Score
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6, edgecolors='w')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, label='Centroids')

    # Set axis labels and title
    plt.title('Fuzzy K-Means Clustering (2D View)')
    plt.xlabel('Age')
    plt.ylabel('Spending Score')

    # Add legend and color bar
    plt.legend()
    plt.colorbar(scatter, label='Cluster Labels')
    plt.show()

    return labels, membership

# Evaluation Function: Silhouette Score and Membership Analysis
def evaluate_clustering(X_scaled, labels, membership):
    # Calculate silhouette score
    sil_score = silhouette_score(X_scaled, labels)
    print(f'Silhouette Score: {sil_score:.2f}')

    # Analyze membership degrees (soft clustering)
    print("\nMembership degrees (top 5 samples):")
    print(membership[:5])  # Print top 5 to avoid large output

# Main function to run the fuzzy k-means clustering and evaluation
def main():
    # Preprocess the data
    X_scaled, X = preprocess_data('Mall_customers.csv')

    # Choose the clustering strategy (Fuzzy C-Means here)
    clustering_strategy = FuzzyCMeansStrategy(n_clusters=3)

    # Run the clustering and 2D visualization
    labels, membership = run_clustering(clustering_strategy, X_scaled, X)

    # Evaluate the clustering result
    evaluate_clustering(X_scaled, labels, membership)

# Execute the main function
if __name__ == "__main__":
    main()

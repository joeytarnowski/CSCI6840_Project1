import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import mixture
from sklearn.metrics import silhouette_score

# Create a Gaussian Mixture Model
def make_gmm(data, x, y, z=None):
    # Use silhouette score to determine the optimal number of components
    sil_scores = []
    for i in range(2, 11):
        gmm = mixture.GaussianMixture(n_components=i)
        if z:
            gmm.fit(data[[x, y, z]])
            labels = gmm.predict(data[[x, y, z]])
            sil_scores.append((i, silhouette_score(data[[x, y, z]], labels)))
        else:
            gmm.fit(data[[x, y]])
            labels = gmm.predict(data[[x, y]])
            sil_scores.append((i, silhouette_score(data[[x, y]], labels)))
        print(f"Silhouette Score for {i} components: {sil_scores[-1][1]}")

    # Select the number of components with the highest silhouette score
    components = max(sil_scores, key=lambda x: x[1])[0]

    # Create the Gaussian Mixture Model
    gmm = mixture.GaussianMixture(n_components=components)
    if z:
        gmm.fit(data[[x, y, z]])
        labels = gmm.predict(data[[x, y, z]])
    else:
        gmm.fit(data[[x, y]])
        labels = gmm.predict(data[[x, y]])

    # Add the labels to the dataframe
    data['cluster'] = labels

    # Create a scatterplot
    if z:
        # 3D plot
        make_gmm_3d(data, components, x, y, z)
    else:
        # 2D plot
        fig, ax = plt.subplots()
        for cluster in range(components):   
            ax.scatter(data[data['cluster'] == cluster][x], data[data['cluster'] == cluster][y], label=cluster)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title('Gaussian Mixture Model')
        ax.legend()
        plt.show()

# Helper function to create 3D scatterplot
def make_gmm_3d(data, components, x, y, z):
    # Create a 3D scatterplot of x, y, and z
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for cluster in range(components):   
        ax.scatter(data[data['cluster'] == cluster][x], data[data['cluster'] == cluster][y], data[data['cluster'] == cluster][z], label=cluster)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title('Gaussian Mixture Model')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    # loading and reading dataset
    df = pd.read_csv("Mall_Customers.csv")
    # Create Gaussian Mixture Models
    # Annual Income and Spending Score
    make_gmm(df, 'Annual Income (k$)', 'Spending Score (1-100)')

    # Age and Spending Score
    make_gmm(df, 'Age', 'Spending Score (1-100)')

    # Age and Annual Income
    make_gmm(df, 'Age', 'Annual Income (k$)')

    # Age, Annual Income, and Spending Score
    make_gmm(df, 'Age', 'Annual Income (k$)', 'Spending Score (1-100)')

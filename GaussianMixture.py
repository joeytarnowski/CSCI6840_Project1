import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
from sklearn.metrics import silhouette_score

path='Pictures/'

# Create a Gaussian Mixture Model
def make_gmm(data, x, y, z=None):
    # Use silhouette score to determine the optimal number of components
    sil_scores = []
    for i in range(2, 11):
        gmm = mixture.GaussianMixture(n_components=i, n_init=50)
        if z:
            gmm.fit(data[[x, y, z]])
            labels = gmm.predict(data[[x, y, z]])
            sil_scores.append((i, silhouette_score(data[[x, y, z]], labels)))
        else:
            gmm.fit(data[[x, y]])
            labels = gmm.predict(data[[x, y]])
            sil_scores.append((i, silhouette_score(data[[x, y]], labels)))

    # Select the number of components with the highest silhouette score
    components = max(sil_scores, key=lambda x: x[1])[0]
    print(f'Optimal number of components: {components}')

    # Create the Gaussian Mixture Model
    gmm = mixture.GaussianMixture(n_components=components)
    if z:
        gmm.fit(data[[x, y, z]])
        labels = gmm.predict(data[[x, y, z]])
    else:
        gmm.fit(data[[x, y]])
        labels = gmm.predict(data[[x, y]])

    # Create a scatterplot with gender and with clusters
    if z:
        print("Silhouette Score: ", silhouette_score(data[[x, y, z]], labels))
        # Show Gender
        data['cluster'] = data['Gender']
        plot = make_gmm_3d(data, components, x, y, z, show_gender=True)
        plot.savefig(f'{path}GMM_{x}_{y}_{z}_gender.png')
        # Show Clusters
        data['cluster'] = labels
        plot = make_gmm_3d(data, components, x, y, z)
        plot.savefig(f'{path}GMM_{x}_{y}_{z}_cluster.png')
    else:
        print("Silhouette Score: ", silhouette_score(data[[x, y]], labels))
        # Show Gender
        data['cluster'] = data['Gender']
        plot = make_2d_gmm(data, components, x, y, show_gender=True)
        plot.savefig(f'{path}GMM_{x}_{y}_gender.png')
        # Show Clusters
        data['cluster'] = labels
        plot = make_2d_gmm(data, components, x, y)
        plot.savefig(f'{path}GMM_{x}_{y}_cluster.png')

def make_2d_gmm(data, components, x, y, show_gender=False):
    # 2D plot
    fig, ax = plt.subplots()
    # Plot the clusters - if show gender is true, make gender the cluster and label the clusters Male and Female
    if show_gender:
        ax.scatter(data[data['cluster'] == 0][x], data[data['cluster'] == 0][y], label="Male")
        ax.scatter(data[data['cluster'] == 1][x], data[data['cluster'] == 1][y], label="Female")
    else:
        for cluster in range(components):   
            ax.scatter(data[data['cluster'] == cluster][x], data[data['cluster'] == cluster][y], label=cluster)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title('Gaussian Mixture Model')
    ax.legend()
    #plt.show()
    return fig


# Helper function to create 3D scatterplot
def make_gmm_3d(data, components, x, y, z, show_gender=False):
    # Create a 3D scatterplot of x, y, and z
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot clusters - if show_gender is true, show the clusters as gender
    if show_gender:
        ax.scatter(data[data['cluster'] == 0][x], data[data['cluster'] == 0][y], data[data['cluster'] == 0][z], label="Male")
        ax.scatter(data[data['cluster'] == 1][x], data[data['cluster'] == 1][y], data[data['cluster'] == 1][z], label="Female")
    else:
        for cluster in range(components):   
            ax.scatter(data[data['cluster'] == cluster][x], data[data['cluster'] == cluster][y], data[data['cluster'] == cluster][z], label=cluster)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title('Gaussian Mixture Model')
    ax.legend()
    plt.show()
    return fig

if __name__ == "__main__":
    # loading and reading dataset
    df = pd.read_csv('Mall_Customers.csv')

    # Convert gender to binary
    df['Gender'].replace(['Male', 'Female'],
                        [0, 1], inplace=True)

    # Create Gaussian Mixture Models
    # Annual Income and Spending Score
    print("\nAnnual Income and Spending Score")
    make_gmm(df, 'Annual Income (k$)', 'Spending Score (1-100)')

    # Age and Spending Score
    print("\nAge and Spending Score")
    make_gmm(df, 'Age', 'Spending Score (1-100)')

    # Age and Annual Income
    print("\nAge and Annual Income")
    make_gmm(df, 'Age', 'Annual Income (k$)')

    # Age, Annual Income, and Spending Score
    print("\nAge, Annual Income, and Spending Score")
    make_gmm(df, 'Age', 'Annual Income (k$)', 'Spending Score (1-100)')

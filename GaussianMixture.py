import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import mixture

# loading and reading dataset
df = pd.read_csv("Mall_Customers.csv")

# create a scatterplot for each numeric column
# plt.figure(figsize=(10,10))
# sns.scatterplot(data=df, x="Annual Income (k$)",y="Spending Score (1-100)", hue="Gender")
# plt.show()

# Fit a Gaussian Mixture Model with five components
gmm = mixture.GaussianMixture(n_components=5)
gmm.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])

# Predict the labels for the data samples
labels = gmm.predict(df[['Annual Income (k$)', 'Spending Score (1-100)']])

# Add the labels to the dataframe
df['cluster'] = labels

# Create a scatterplot of the data
fig, ax = plt.subplots()
for cluster in range(5):   
    ax.scatter(df[df['cluster'] == cluster]['Annual Income (k$)'], df[df['cluster'] == cluster]['Spending Score (1-100)'], label=cluster)
ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Spending Score (1-100)')
ax.set_title('Gaussian Mixture Model')
ax.legend()
plt.show()

# Print the cluster centers
print(gmm.means_)
print(gmm.covariances_)
print(gmm.weights_)
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd

# Ensure that the file is in the same directory
filename = "Mall_Customers.csv" 

# load data into dataframe
df = pd.read_csv(filename)

# Set for Annual Income and Spending Score Columns
data=df.iloc[:,[3,4]].values

wcss=[] # within cluster sum of square
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++',random_state=0)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_) #inertia_ = to find the wcss value

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

### KMEANS Annual Income, Spending Score ###

# Select Items
data = df.iloc[:,[3,4]].values

# Run KMeans
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0)
y_kmeans = kmeans.fit_predict(data)

# Silhouette Score of Clusters: Analysis of Clusters
silhouette_avg = silhouette_score(data, y_kmeans)
print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

# Plot Clusters
fig,ax = plt.subplots(figsize=(14,6))
ax.scatter(data[y_kmeans==0,0],data[y_kmeans==0,1],s=100,c='blue',label='Cluster 1')
ax.scatter(data[y_kmeans==1,0],data[y_kmeans==1,1],s=100,c='red',label='Cluster 2')
ax.scatter(data[y_kmeans==2,0],data[y_kmeans==2,1],s=100,c='green',label='Cluster 3')
ax.scatter(data[y_kmeans==3,0],data[y_kmeans==3,1],s=100,c='magenta',label='Cluster 4')
ax.scatter(data[y_kmeans==4,0],data[y_kmeans==4,1],s=100,c='cyan',label='Cluster 5')

# Add Centroids and Graph Titles
ax.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=400,c='yellow',label='Centroid')
plt.title('Cluster Segmentation of Customers')
plt.xlabel('Annual Income(K$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()

### KMeans Age, Annual Income ### 

data=df.iloc[:,[2,3]].values

data = df.iloc[:,[2,4]].values
wcss=[]  # within cluster sum of square
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++',random_state=0)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)  # inertia_ = to find the wcss value

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans=KMeans(n_clusters=4,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(data)

#Plotting the clusters
fig,ax = plt.subplots(figsize=(14,6))
ax.scatter(data[y_kmeans==0,0],data[y_kmeans==0,1],s=100,c='red',label='Cluster 1')
ax.scatter(data[y_kmeans==1,0],data[y_kmeans==1,1],s=100,c='blue',label='Cluster 2')
ax.scatter(data[y_kmeans==2,0],data[y_kmeans==2,1],s=100,c='green',label='Cluster 3')
ax.scatter(data[y_kmeans==3,0],data[y_kmeans==3,1],s=100,c='cyan',label='Cluster 4')

ax.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=400,c='yellow',label='Centroid')
plt.title('Cluster Segmentation of Customers')
plt.xlabel('Age')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()

# Silhouette Score of Clusters: Analysis of Clusters
silhouette_avg = silhouette_score(data, y_kmeans)
print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
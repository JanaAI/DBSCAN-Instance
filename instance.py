import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Read dataset from CSV
df = pd.read_csv('for_interns.csv')

X = df[['clusters', 'src_port']].values

# Compute DBSCAN
dbscan = DBSCAN(eps=55, min_samples=2)
clusters = dbscan.fit_predict(X)

# Count instances in each cluster
unique_labels, counts = np.unique(clusters[clusters != -1], return_counts=True)
unique_clusters, counts_all = np.unique(clusters, return_counts=True)

# Print number of instances in each cluster
for cluster_id, count in zip(unique_clusters, counts_all):
    print(f"Cluster {cluster_id}: {count} instances")


    # Plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clustering')
plt.show()


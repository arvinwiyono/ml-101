import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, 3:5].values

# Plot the dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
plt.figure()
dendogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidian distances')
plt.show()

# From the dendrogram
# Take the left most vertical line, and therefore we get 5 as the optimal number of clusters

from sklearn.cluster import AgglomerativeClustering
agc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
clusters = agc.fit_predict(x)

# Visualising the clusters
plt.figure()
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (K$)')
plt.ylabel('Spending Score (1-100)')
for label in np.unique(agc.labels_):
    plt.scatter(x[clusters == label, 0], x[clusters == label, 1], color = plt.get_cmap('tab10')(label), label = f"Cluster {label}")
plt.legend()
plt.show()
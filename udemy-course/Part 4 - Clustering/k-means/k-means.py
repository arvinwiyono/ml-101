# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
#x = dataset.iloc[:, 1:5].values
#
#from sklearn.preprocessing import LabelEncoder
#label_enc = LabelEncoder()
#x[:, 0] = label_enc.fit_transform(x[:, 0])
#x = x.astype(int)
#
## Find out the optimal number of clusters
from sklearn.cluster import KMeans
#wcss = []
#for i in range(1, 11):
#    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
#    kmeans.fit(x)
#    wcss.append(kmeans.inertia_)
#    
#plt.plot(range(1,11), wcss)
#plt.title('The Elbow Method')
#plt.xlabel('Number of Clusters')
#plt.ylabel('WCSS')
#
## the most optimal K means seem to be 8
#kmeans = KMeans(n_clusters = 8, random_state = 0)
#clusters = kmeans.fit_predict(x)

x = dataset.iloc[:, 3:5].values
wcss = list()
for i in range(1,11):
    wcss.append(KMeans(n_clusters = i).fit(x).inertia_)

plt.figure()
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS') 

# From here 5 seems to be the optimal number of clusters
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
clusters = kmeans.fit_predict(x)

# Visualising the results
plt.figure()
colors = ['red', 'green', 'blue', 'orange', 'purple']
for i in np.unique(kmeans.labels_):
    plt.scatter(x[clusters == i, 0], x[clusters == i, 1], s = 20, c = plt.get_cmap('tab10').colors[i], label = 'Cluster ' + str(i))
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 80, c = 'black', label = 'Centroids')
plt.legend()
plt.xlabel('Annual Income (K$)')
plt.ylabel('Spending Score (1-100)')
plt.savefig('clusters.png', format='png')
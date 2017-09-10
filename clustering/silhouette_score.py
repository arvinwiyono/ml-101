import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
X = np.c_[x1, x2]

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']
num_clusters = [2, 3, 4, 5, 8]
wcss = []

for i in range(1, 7):
    plt.subplot(3,2,i)
    plt.tight_layout()
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.scatter(x1, x2)
    plt.title('Instances')
    if i > 1:
        index = i - 2
        kmeans_model = KMeans(n_clusters=num_clusters[index], random_state=42).fit(X)
        plt.title(f"{num_clusters[index]} cluster(s)")
        plt.xlabel("silhouette coefficient = {:.3f}".format(metrics.silhouette_score(X, kmeans_model.labels_, metric='euclidean')))
        for j, label in enumerate(kmeans_model.labels_):
            plt.scatter(x1[j], x2[j], color=colors[label], marker=markers[label])
        wcss.append(kmeans_model.inertia_)
plt.show()
plt.close()
plt.figure(num=1)
plt.title('Within-Cluster Sum of Squares')
plt.plot(num_clusters, wcss)
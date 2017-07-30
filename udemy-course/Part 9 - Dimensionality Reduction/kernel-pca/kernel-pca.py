# Dimensionality Reduction: Kernel-PCA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Perform kernelized-PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=2, kernel='rbf')
x_train = kpca.fit_transform(x_train)
x_test = kpca.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

# Predicting test dataset
y_pred = classifier.predict(x_test)

score_train = classifier.score(x_train, y_train)
score_test = classifier.score(x_test, y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising train results
x_set, y_set = x_train, y_train
pc1_range = np.arange(x_set[:, 0].min()-1, x_set[:, 0].max()+1, 0.01)
pc2_range = np.arange(x_set[:, 1].min()-1, x_set[:, 1].max()+1, 0.01)

x1, x2 = np.meshgrid(pc1_range, pc2_range)
data_points = np.array([x1.ravel(), x2.ravel()]).T
pred_data = classifier.predict(data_points).reshape(x1.shape)
plt.figure()
plt.title('Logistic Regression - Train Data')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.contourf(x1, x2, pred_data, alpha = 0.1, cmap = plt.get_cmap('Dark2'))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = plt.get_cmap('Dark2')(i), label = f"Customer Segment {j}")
plt.legend()
plt.show()


# Visualising train results
x_set, y_set = x_test, y_test
pc1_range = np.arange(x_set[:, 0].min()-1, x_set[:, 0].max()+1, 0.01)
pc2_range = np.arange(x_set[:, 1].min()-1, x_set[:, 1].max()+1, 0.01)

x1, x2 = np.meshgrid(pc1_range, pc2_range)
data_points = np.array([x1.ravel(), x2.ravel()]).T
pred_data = classifier.predict(data_points).reshape(x1.shape)
plt.figure()
plt.title('Logistic Regression - Test Data')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.contourf(x1, x2, pred_data, alpha = 0.1, cmap = plt.get_cmap('Dark2'))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = plt.get_cmap('Dark2')(i), label = f"Customer Segment {j}")
plt.legend()
plt.show()
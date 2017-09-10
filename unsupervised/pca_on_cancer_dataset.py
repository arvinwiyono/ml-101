from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

breast_cancer = load_breast_cancer()
x = breast_cancer.data
y = breast_cancer.target

# This takes time, comment if not needed
malignant = x[y == 0]
benign = x[y == 1]

fig, axes = plt.subplots(15, 2, figsize=(6, 30))
axes = axes.ravel()
fig.tight_layout()
fig.subplots_adjust(hspace=0.8)
for i in range(30):
    _, bins = np.histogram(x[:, i], bins=50)
    axes[i].hist(malignant[:, i], bins=bins, color='b', alpha=.5)
    axes[i].hist(benign[:, i], bins=bins, color='r', alpha=.5)
    axes[i].set_title(breast_cancer.feature_names[i])
fig.suptitle('Breast Cancer Histogram', y=1.01)
plt.show()
#####

scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

pca = PCA(n_components=2)
pca.fit(x_scaled)
x_pca = pca.transform(x_scaled)
print(f"Original shape: {x.shape}")
print(f"Reduced shape: {x_pca.shape}")

colors = plt.get_cmap('Set1').colors[:2]
plt.figure(num=1, figsize=(7,7))
for target in np.unique(y):
    plt.scatter(x_pca[y==target, 0], x_pca[y==target, 1], c=colors[target], edgecolor='k', alpha=.7, label=breast_cancer.target_names[target])
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.gca().set_aspect('equal')
plt.title('Reduced Breast Cancer Features')
plt.legend()
plt.show()

# visualise pca components
plt.figure(num=1)
plt.matshow(pca.components_, cmap='viridis_r')
plt.yticks([0, 1], ['first_component', 'second component'])
plt.colorbar()
plt.xticks(range(len(breast_cancer.feature_names)), breast_cancer.feature_names, rotation=60)
plt.show()
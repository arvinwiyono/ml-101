# Classification Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, 2:-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# import the sklearn library

# predicting test data
y_pred = classifier.predict(x_test)

# evaluating results
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# visualising train results
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
age_range = np.arange(x_set[:, 0].min()-1, x_set[:, 0].max()+1, 0.01)
salary_range = np.arange(x_set[:, 1].min()-1, x_set[:, 1].max()+1, 0.01)

x1, x2 = np.meshgrid(age_range, salary_range)
# numpy ravel flattens the array
# .T transposes the array
data_points = np.array([x1.ravel(), x2.ravel()]).T # data points for contouring
pred_data = classifier.predict(data_points).reshape(x1.shape)

color_map = ListedColormap(('red', 'green'))

plt.contourf(x1, x2, pred_data, alpha = 0.3, cmap = color_map)
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

labels = ['Not Purchased', 'Purchased']
# plotting the real data
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = color_map.colors[j], label = labels[j])

plt.title('Logistic Regression (Train Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# visualising test results
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
age_range = np.arange(x_set[:, 0].min()-1, x_set[:, 0].max()+1, 0.01)
salary_range = np.arange(x_set[:, 1].min()-1, x_set[:, 1].max()+1, 0.01)

x1, x2 = np.meshgrid(age_range, salary_range)
data_points = np.array([x1.ravel(), x2.ravel()]).T
pred_data = classifier.predict(data_points).reshape(x1.shape)

color_map = ListedColormap(('red', 'green'))

plt.contourf(x1, x2, pred_data, alpha = 0.3, cmap = color_map)
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
labels = ['Not Purchased', 'Purchased']
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = color_map.colors[j], label = labels[j])

plt.title('Logistic Regression (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
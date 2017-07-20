# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
scaler_y = StandardScaler()

scaled_x = scaler_x.fit_transform(x)
scaled_y = scaler_y.fit_transform(y)

"""
WARN: SVR does not apply feature scaling automatically
"""
from sklearn.svm import SVR
reg = SVR(kernel='rbf')
reg.fit(scaled_x, scaled_y)

# test with a value
y_pred = reg.predict(scaler_x.transform(np.array([[6.5]])))
print(scaler_y.inverse_transform([y_pred]))

# Visualising SVR results
plt.figure()
plt.scatter(scaled_x, scaled_y, color='purple')
plt.plot(scaled_x, reg.predict(scaled_x))
plt.title('SVR')
plt.xlabel('Position Level')
plt.ylabel('Salary ($)')

# With a smoother curve
plt.figure()
x_grid = np.arange(min(scaled_x), max(scaled_x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(scaled_x, scaled_y, color='purple')
plt.plot(x_grid, reg.predict(x_grid))
plt.title('SVR (More Data Points)')
plt.xlabel('Position Level')
plt.ylabel('Salary ($)')
# Random Forest Regression
# Ensemble learning comprises n_estimators of base estimators
# In the case of random forest, the base estimator is a decision tree

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting the Regression Model to the dataset
# Create your regressor here
from sklearn.ensemble import RandomForestRegressor
# n_estimators = the number of trees
# random_state for reproducibility of results
regressor = RandomForestRegressor(n_estimators=1000, max_features = 1, random_state=0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

test_data = np.arange(min(X), max(X), 0.5)
test_data = test_data.reshape((len(test_data), 1))
test_result = regressor.predict(test_data)
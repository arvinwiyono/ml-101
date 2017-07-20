"""
Polynomial Regression

@author: Arvin Wiyono
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# no need to conern about position name as level can be the predictor
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression

# build linear regression model
lin_rg = LinearRegression()
lin_rg.fit(x, y)

# build polynomial regression model to compare
from sklearn.preprocessing import PolynomialFeatures #this adds x1^2, x1^3, x1^4, etc.
poly_adder = PolynomialFeatures(degree=4)
x_poly = poly_adder.fit_transform(x)

poly_rg = LinearRegression()
poly_rg.fit(x_poly, y)

# visualising linear regression results
plt.figure(dpi=100)
#plt.subplot(2, 1, 1)
plt.title('Linear Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.scatter(x, y, color='pink')
plt.plot(x, lin_rg.predict(x), color='green')
#plt.savefig('linear.png', format='png')

# visualising polynomial regression results
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)

plt.figure(dpi=100)
#plt.subplot(2, 1, 2)
plt.title('Polynomial Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.scatter(x, y, color='pink')
plt.plot(x_grid, poly_rg.predict(poly_adder.fit_transform(x_grid)), color='green')
#plt.savefig('polynomial.png', format='png')
#plt.show()

# try predicting
print(lin_rg.predict(6.5)[0])

print(poly_rg.predict(poly_adder.fit_transform(6.5))[0])
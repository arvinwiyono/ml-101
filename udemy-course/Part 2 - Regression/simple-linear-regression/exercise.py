import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,:-1].values # needs to be a matrix
Y = dataset['Salary'].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1.0/3, random_state=0)

# data is well preprocessed now

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#from sklearn.model_selection import cross_val_score
#regressor_cv = LinearRegression()
#cross_val_score(regressor_cv, X_train, Y_train, cv=4)

# Predicting the test results
Y_pred = regressor.predict(X_test)
coeff_r2 = regressor.score(X_test, Y_test)
print("Coefficient R2 is {0:>5.5f}".format(coeff_r2))

# Visualising the training set results
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='green')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.savefig('salary-vs-experience-training.png', format='png', dpi=350)
plt.show()

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='green')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.savefig('salary-vs-experience-testing.png', format='png', dpi=350)
plt.show()
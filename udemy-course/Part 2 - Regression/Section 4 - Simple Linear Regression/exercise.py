import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,:-1].values # needs to be a matrix
Y = dataset['Salary'].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1.0/3, random_state=42)

# data is well preprocessed now

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#from sklearn.model_selection import cross_val_score
#regressor_cv = LinearRegression()
#cross_val_score(regressor_cv, X_train, Y_train, cv=4)


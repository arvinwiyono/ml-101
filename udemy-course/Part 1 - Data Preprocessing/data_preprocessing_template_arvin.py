import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
data = pd.read_csv('Data.csv')

# experiment
print(data.iloc[1:4, :-2]) # this returns the dataframe
print(data.iloc[1:4, :-2].values) # this returns a numpy array
print(type(data.iloc[1:4, :-2].values))

# separate the independent variables (features)
X = data.iloc[:, :-1].values
print(X)

# the labels
Y = data.iloc[:, -1].values
print(Y)

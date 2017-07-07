# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Categorical variables are: Country and Purchased
from sklearn.preprocessing import LabelEncoder
le_X = LabelEncoder()
X[:, 0] = le_X.fit_transform(X[:, 0])

# however this leads to another problem
# the machine learning model can think that Spain is higher than France,
# Germany is smaller than Spain, etc.
# So there should not be a relational order

# A better way is using one hot encoding or dummy encoding
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(categorical_features = [0]) # specify the column that we one-hot-encode
X = enc.fit_transform(X).toarray() # can only take numerical inputs

# For the Purchased, we just need to use the Label Encoder as it is only either 0 or 1
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Splitting the dataset into the Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
# Put the values of age and salary on the same scale
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
scaled_X_train = sc_X.fit_transform(X)
# already fitted, so just use transform()
scaled_X_test = sc_X.transform(X_test)

# Or you can scale only the age and salary exclusively
# X_train[:, 3:5] = sc_X.fit_transform(X_train[:, 3:5])

from sklearn.preprocessing import RobustScaler
rbsc_X = StandardScaler()
rbsc_X_train = rbsc_X.fit_transform(X_train)
# already fitted, so just use transform()
rbsc_X_test = rbsc_X.transform(X_test)

from sklearn.preprocessing import MinMaxScaler
min_max_sc_X = MinMaxScaler()
min_max_X_train = min_max_sc_X.fit_transform(X_train)
# already fitted, so just use transform()
min_max_X_test = min_max_sc_X.transform(X_test)


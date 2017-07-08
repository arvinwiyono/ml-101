# Multivariate Linear Regression with Backward Elimination

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values # the last column (Profit)

# Data encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X[:, -1] = le.fit_transform(X[:,-1]) # labelling the State
enc = OneHotEncoder(categorical_features=[3])
X = enc.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:, 1:] # ignoring the first dummy column :)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

score = regressor.score(X_test, y_test)
print(f"Score is {score}")

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print('Predicted vs Real errors:', mae, '||', mse)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as smf

# this adds the ones at the end, but we want to put it at the beginning
#X = np.append(arr=X, values = np.ones((len(X),1)), axis = 1)
# this adds the ones at the beginning
X = np.append(arr = np.ones((50,1)), values = X, axis = 1)

# Start Backward Elimination
X_opt = X[:,[0,1,2,3,4,5]] # original optimal matrix
regressor_ols = smf.OLS(endog=y, exog=X_opt).fit()
print(regressor_ols.summary())

# Remove first dummy var, which has a p-val = 0.990 > 0.05
X_opt = X[:,[0,1,3,4,5]] # original optimal matrix
regressor_ols = smf.OLS(endog=y, exog=X_opt).fit()
print(regressor_ols.summary())
print("\n")

# Remove second dummy var, which has a p-val = 0.940 > 0.05
X_opt = X[:,[0,3,4,5]] # original optimal matrix
regressor_ols = smf.OLS(endog=y, exog=X_opt).fit()
print(regressor_ols.summary())
print("\n")

# Remove Administration, which has a p-val = 0.602 > 0.05
X_opt = X[:,[0,3,5]] # original optimal matrix
regressor_ols = smf.OLS(endog=y, exog=X_opt).fit()
print(regressor_ols.summary())
print("\n")

# Remove Marketing Spend, which has a p-val = 0.06 > 0.05
X_opt = X[:,[0,3]] # original optimal matrix
regressor_ols = smf.OLS(endog=y, exog=X_opt).fit()
print(regressor_ols.summary())
print("\n")

# Therefore, in this case, the significant predictor is only the R&D spend
# Now build the optimal regressor
print("------------OPTIMAL MODEL---------------")
X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size=0.2, random_state=0)
opt_regressor = LinearRegression()
opt_regressor.fit(X_train, y_train)
y_pred_opt = opt_regressor.predict(X_test)
score = opt_regressor.score(X_test, y_test)
print(f"Score is {score}")

mae_opt = mean_absolute_error(y_test, y_pred_opt)
mse_opt = mean_squared_error(y_test, y_pred_opt)
# mae_opt and mse_opt should be lower than original mae and mse
print('Predicted vs Real errors:', mae_opt, '||', mse_opt) 
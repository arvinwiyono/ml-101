#==============================================================================
# Arvin
#==============================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
# for handling missing data
# documentation: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html#sklearn.preprocessing.Imputer

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

# axis = 0 means applying the strategy on the column
# in this case, the mean of the column Age & Salary
imputer = Imputer(missing_values="NaN",
                  strategy="median",
                  axis=0)
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

for data in X:
    print(data)
    
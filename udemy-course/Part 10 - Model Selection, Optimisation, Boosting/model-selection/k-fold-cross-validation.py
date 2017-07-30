# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 00:45:39 2017
k-Fold Cross Validation
@author: Arvin
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Define classifier here
classifier = SVC(random_state=0)
classifier.fit(x_train, y_train)

# Predict test data
y_pred= classifier.predict(x_test)

# Evaluation
cm = confusion_matrix(y_test, y_pred)
score = classifier.score(x_test, y_test)

# Applying cross validation here
accuracies = cross_val_score(classifier, sc.transform(x), y, cv=10)
print(accuracies.mean())
print(accuracies.std())

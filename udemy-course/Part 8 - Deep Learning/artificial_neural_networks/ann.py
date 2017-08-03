"""
Artificial Neural Network

Keras - pip install keras
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PART 1 - Data Preprocessing
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_country = LabelEncoder()
label_encoder_gender = LabelEncoder()

# Encode country and gender
x[:, 1] = label_encoder_country.fit_transform(x[:, 1])
x[:, 2] = label_encoder_gender.fit_transform(x[:, 2])

ohe = OneHotEncoder(categorical_features=[1])
x = ohe.fit_transform(x).toarray()
# Avoid dummy variable trap
x = x[:, 1:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Scaling is mandatory
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# PART 2 - Make the ANN
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
# This is the input layer
# (11 features + 1) / 2 = 6 nodes
# Input dimension = 11 as we have 11 features
classifier.add(Dense(6, activation='relu', input_dim=11))

# Second and third layer
classifier.add(Dense(6, activation='relu'))
classifier.add(Dense(6, activation='relu'))

# Add the final output layer with sigmoid activation function
classifier.add(Dense(1, activation='sigmoid'))

# Compile the brain
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the ANN to the training set
classifier.fit(x_train, y_train, batch_size=32, epochs=100)

# PART 3 - Predict!
y_pred = classifier.predict_classes(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
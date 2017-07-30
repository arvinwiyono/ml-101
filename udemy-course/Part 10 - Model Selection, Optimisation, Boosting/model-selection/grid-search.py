# Grid Search

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

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

# Perform grid search
parameters = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': np.arange(0.1, 1.0, 0.1)}
        ]
svc = SVC()
"""
Valid options are ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']
"""
grid_search = GridSearchCV(svc, parameters, scoring='accuracy', cv=10)
grid_search.fit(x_train, y_train)
results = grid_search.cv_results_
best_accuracy = grid_search.best_score_
best_params = grid_search.best_params_
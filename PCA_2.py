# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 20:51:46 2023

@author: JPMM
"""




from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder



# Load the iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns = iris.feature_names)
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate a PCA object with 2 components
pca = PCA(n_components=1)

# Fit the PCA model to the training data
X_train_pca = pca.fit_transform(X_train)

# Train a logistic regression classifier on the transformed data
clf = LogisticRegression(random_state=42)
clf.fit(X_train_pca, y_train)

# Apply the same PCA transformation to the testing data
X_test_pca = pca.transform(X_test)

# Make predictions on the testing data
y_pred = clf.predict(X_test_pca)

# Evaluate the accuracy of the predictions
accuracy = round(accuracy_score(y_test, y_pred), 7)
print(f"Accuracy: {accuracy}")






rf = RandomForestRegressor(n_estimators=150)
rf.fit(X_train, y_train)
sort = rf.feature_importances_.argsort()
plt.barh(y = np.array(data.columns)[sort], width = rf.feature_importances_[sort])

plt.xlabel("Feature Importance")
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 18:03:56 2023

@author: JPMM
"""



#   https://www.youtube.com/watch?v=8klqIM9UvAc




import pandas as pd
from sklearn.datasets import load_digits
import numpy as np

dataset = load_digits()
dataset.keys()

from matplotlib import pyplot as plt

plt.gray()
plt.matshow(dataset.data[233].reshape(8,8))

df = pd.DataFrame(dataset.data, columns = dataset.feature_names)

df.describe()

X = df
y = dataset.target

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split (X_scaled, y, test_size = 0.2, random_state = 30)
model = LogisticRegression(max_iter = 1000)
model.fit (X_train, y_train)
print (model.score (X_test, y_test))




from sklearn.decomposition import PCA

pca = PCA(0.95)
pca = PCA(n_components = 50)

X_pca = pca.fit_transform(X)

X_pca.shape
pca.explained_variance_ratio_
pca.n_components

X_train, X_test, y_train, y_test = train_test_split (X_pca, y, test_size = 0.2, random_state = 30)

model = LogisticRegression(max_iter = 1000)

model.fit (X_train, y_train)
print (model.score (X_test, y_test))



# -*- coding: utf-8 -*- ''''' JP
"""
Created on Wed Feb 15 17:56:02 2023

@author: JPMM
"""



import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data = pd.read_excel('data.xlsx',sheet_name = 'data')
data.dropna(inplace=True)

for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col])   



X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])


X = data.values
pca = PCA(n_components=5)
pca.fit(X)
#PCA(n_components=1)
print(pca.explained_variance_ratio_)

print(pca.singular_values_)





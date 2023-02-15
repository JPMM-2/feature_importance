# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 10:57:31 2023

@author: JPMM
"""


import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()
data = pd.read_excel('data.xlsx',sheet_name = 'data')
data.dropna(inplace=True)

for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col])        

X = data.drop(columns = ['nota5'])
y = data['nota5']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

rf = RandomForestRegressor(n_estimators=150)
rf.fit(X_train, y_train)
sort = rf.feature_importances_.argsort()
plt.barh(y = np.array(data.columns)[sort], width = rf.feature_importances_[sort])

plt.xlabel("Feature Importance")






#=================================================================
#=================================================================


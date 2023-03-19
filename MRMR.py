

# import libraries
import pandas as pd
from skfeature.function.information_theoretical_based import MRMR

import seaborn as sns

# load titanic dataset from seaborn
titanic0 = sns.load_dataset('titanic')

titanic = pd.get_dummies(titanic0)

# separate features and target variable
X = titanic.drop(['survived'], axis=1)
y = titanic['survived']

# perform MRMR feature selection
n_features = 4  # choose number of features to select
selected_features = MRMR.mrmr(X.values, y.values, n_selected_features=n_features)

# subset the dataset with selected features
X_selected = X.iloc[:, selected_features]

# print selected feature names
print(X_selected.columns)








# import necessary libraries
import pandas as pd

# create example dataframe with categorical variable
df = pd.DataFrame({'color': ['red', 'blue', 'green', 'green', 'red', 'blue', 'blue']})

# perform one-hot encoding
one_hot_df = pd.get_dummies(df['color'])

# concatenate original dataframe with one-hot encoded dataframe
df = pd.concat([df, one_hot_df], axis=1)

# drop original categorical variable
#df = df.drop('color', axis=1)

# print resulting dataframe
print(df)


















import warnings
warnings.filterwarnings('ignore')

# -----------
# Import data
# -----------

import pandas as pd

file_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
df = pd.read_csv(file_path, header=None)
column_heads = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', '#doors', 'body-style', 
'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
'#cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rmp', 
'city-mpg', 'highway-mpg', 'price']
df.columns = column_heads

# print('Imported Data:\n{}'.format(df.dtypes))

# ----------------
# Cleaning up data
# ----------------

from numpy import nan, mean, meshgrid, array

df['price'].replace('?', nan, inplace=True)
df.dropna(subset=['price'], axis=0, inplace=True)
df['price'] = df['price'].astype('float')

df['horsepower'].replace('?', nan, inplace=True)
df['horsepower'] = df['horsepower'].astype('float')
df_horsepower_mean = mean(df['horsepower'])
df['horsepower'].replace(nan, df_horsepower_mean, inplace=True)

# -----------
# Grid Search
# -----------

X = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
Y = df['price']

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

param1  = [
    {'alpha': [0.001, 0.01, 0.1, 1, 10, 100] }, 
    {'normalize': [True, False]}
]
rr = Ridge()
grid1 = GridSearchCV(rr, param1, cv=4)
grid1.fit(X, Y)
print(grid1.best_estimator_)
# print(grid1.cv_results_['mean_test_score'])

scores = grid1.cv_results_
for param, mean_score in zip(scores['params'], scores['mean_test_score']):
    print('Param: {}, R^2 (Test): {}'.format(param, mean_score))

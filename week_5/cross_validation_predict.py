# -----------
# Import data
# -----------

import pandas as pd
from pandasgui import show 

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

from numpy import nan, mean

df['price'].replace('?', nan, inplace=True)
df.dropna(subset=['price'], axis=0, inplace=True)
df['price'] = df['price'].astype('float')

df['horsepower'].replace('?', nan, inplace=True)
df['horsepower'] = df['horsepower'].astype('float')
df_horsepower_mean = mean(df['horsepower'])
df['horsepower'].replace(nan, df_horsepower_mean, inplace=True)

# show(df)

# -----------------------
# Cross Validation Preict
# -----------------------

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

mlr = LinearRegression()
X = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
Y = df['price']
mlr.fit(X, Y)
Y_hat = cross_val_predict(mlr, X, Y, cv=4)
print(Y_hat)

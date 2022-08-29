# -----------
# Import Data
# -----------

import pandas as pd

file_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
df = pd.read_csv(file_path, header=None)
column_heads = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', '#doors', 'body-style', 
'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
'#cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rmp', 
'city-mpg', 'highway-mpg', 'price']
df.columns = column_heads

# ------------
# Data Cleanup
# ------------

import numpy as np

df['price'].replace('?', np.nan, inplace=True)
df.dropna(subset=['price'], axis=0, inplace=True)
df['price'] = df['price'].astype('float')

df['horsepower'].replace('?', np.nan, inplace=True)
mean_horsepower = df['horsepower'].astype('float').mean(axis=0)
df['horsepower'].replace(np.nan, mean_horsepower, inplace=True)
df['horsepower'] = df['horsepower'].astype('float')

# ----------------
# Train Test Split
# ----------------

from sklearn.model_selection import train_test_split
from pandasgui import show

X = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
show(x_train, x_test, y_train, y_test)
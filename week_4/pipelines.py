# ---------------------------
# Importing Training Data-Set
# ---------------------------

from turtle import color
import pandas as pd

file_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
df = pd.read_csv(file_path, header=None)
column_heads = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', '#doors', 'body-style', 
'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
'#cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rmp', 
'city-mpg', 'highway-mpg', 'price']
df.columns = column_heads

# -----------------------------
# Cleaning up Imported Data-Set
# -----------------------------

import numpy as np

df['price'].replace('?', np.nan, inplace=True)
df.dropna(subset=['price'], axis=0, inplace=True)
df['price'] = df['price'].astype('float')

df['horsepower'].replace('?', np.nan, inplace=True)
mean_horsepower = df['horsepower'].astype('float').mean(axis=0)
df['horsepower'].replace(np.nan, mean_horsepower, inplace=True)
df['horsepower'] = df['horsepower'].astype('float')

# --------------------------
# Generating Pipeline object
# --------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline

process_list = [('scale', StandardScaler()), ('poly', PolynomialFeatures(degree=2, include_bias=False)), ('mode', LinearRegression())]
pipe = Pipeline(process_list)

# --------------------------------------------------------
# Training the Pipeline object using the training data-set
# --------------------------------------------------------

Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
Y = df['price']
pipe.fit(Z, Y)

# --------------------------------------
# Plot the Actual vs. Predicted data-set
# --------------------------------------

X = range(0, len(df.index), 1)
Y_hat = pipe.predict(Z)

from matplotlib import pyplot as plt

plt.figure(0)
plt.scatter(X, Y, color='r')
plt.scatter(X, Y_hat, color='b')
plt.savefig('./pipeline_actual_vs_predicted.png')
plt.show()
plt.close(0)
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

# --------------------------
# Multiple Linear Regression
# --------------------------

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
Y = df['price']
lm.fit(Z, Y)
print('Multiple Linear Regression(MLR) Model: Intercept(b0): {} | Coefficients: {}'.format(lm.intercept_, lm.coef_))

# ---------------------------
# R^2, MSE Calculation of MLR
# ---------------------------

R_squared = lm.score(Z, Y)
print('R^2 Score of SLR: {:.4f}'.format(R_squared))

from sklearn.metrics import mean_squared_error

Y_hat = lm.predict(Z)
MSE = mean_squared_error(Y, Y_hat)
print('Mean Squared Error (MSE) of SLR: {:.2f}'.format(MSE))

# -----------------------------
# MLR Actual vs. Predicted Plot
# -----------------------------

from matplotlib import pyplot as plt

X = range(1, len(df.index)+1, 1)
plt.figure(0)
plt.scatter(X, Y, color='r')
plt.scatter(X, Y_hat, color='b')
plt.savefig('./mlr_actual_vs_predicted.png')
plt.show()



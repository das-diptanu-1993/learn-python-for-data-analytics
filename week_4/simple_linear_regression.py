# -----------
# Import Data
# -----------

from turtle import color
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

# print('Initial data types of the columns: \n', df.dtypes)
df['price'].replace('?', np.nan, inplace=True)
mean_price = df['price'].astype('float').mean(axis=0)
df['price'].replace(np.nan, mean_price, inplace=True)
df['price'] = df['price'].astype('float')
# print('\n\nModified data types of the columns: \n', df.dtypes)

# ------------------------
# Simple Linear Regression
# ------------------------

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
X = df[['highway-mpg']]
Y = df['price']
lm.fit(X, Y)
print('Linear Regression Model: Intercept(b0): {} | Slope(b1): {}'.format(lm.intercept_, lm.coef_))

# ---------------------------
# R^2, MSE Calculation of SLR
# ---------------------------

R_squared = lm.score(X, Y)
print('R^2 Score of SLR: {:.4f}'.format(R_squared))

from sklearn.metrics import mean_squared_error

Y_hat = lm.predict(X)
MSE = mean_squared_error(Y, Y_hat)
print('Mean Squared Error (MSE) of SLR: {:.2f}'.format(MSE))

# -------------------------------
# SLR Plot - Actual vs. Predicted
# -------------------------------

from matplotlib import pyplot as plt

plt.figure(0)
plt.scatter(X, Y, color='red')
plt.scatter(X, Y_hat, color='blue')
plt.savefig('./slr_actual_vs_predicted.png')
plt.show()
plt.close(0)



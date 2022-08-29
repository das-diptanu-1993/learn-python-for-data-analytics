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

# ----------------------------------------------
# Polynomial Regression with Multiple Dimensions
# ----------------------------------------------

from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree=2, include_bias=False)
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
Z_pr = pr.fit_transform(Z)


from sklearn.linear_model import LinearRegression

lm = LinearRegression()
Y = df['price']
lm.fit(Z_pr, Y)

# ---------------------------
# R^2, MSE Calculation of MPR
# ---------------------------

R_squared = lm.score(Z_pr, Y)
print('R^2 Score of SLR: {:.4f}'.format(R_squared))

from sklearn.metrics import mean_squared_error

Y_hat = lm.predict(Z_pr)
MSE = mean_squared_error(Y, Y_hat)
print('Mean Squared Error (MSE) of SLR: {:.2f}'.format(MSE))

# -----------------------------
# MPR Actual vs. Predicted Plot
# -----------------------------

from matplotlib import pyplot as plt

plt.figure(0)
X = range(0, len(df.index), 1)
plt.scatter(X, Y, color='r')
plt.scatter(X, Y_hat, color='b')
plt.savefig('./mpr_actual_vs_predicted.png')
plt.show()
plt.close(0)

# ---------------------
# MPR Distribution Plot
# ---------------------

import seaborn as sns

plt.figure(1)
ax1 = sns.distplot(Y, hist=False, color='r', label='Actual')
sns.distplot(Y_hat, hist=False, color='b', label='Fitted', ax=ax1)
plt.savefig('./mpr_distribution_plot.png')
plt.show()
plt.close(1)






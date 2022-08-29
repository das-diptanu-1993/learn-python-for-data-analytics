# -----------
# Import Data
# -----------

from cProfile import label
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

# -----------------
# Distribution Plot
# -----------------

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(df[['highway-mpg']], df['price'])
Y_hat = lm.predict(df[['highway-mpg']])

import seaborn as sns
from matplotlib import pyplot as plt

ax1 = sns.distplot(df['price'], hist=False, color='r', label='Actual')
sns.distplot(Y_hat, hist=False, color='b', label='Fitted', ax=ax1)

plt.savefig('./distribution_plot.png')
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
df = pd.read_csv(path, header=None)

column_headers = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', '#doors', 'body-style', 
'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
'#cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rmp', 
'city-mpg', 'highway-mpg', 'price']
df.columns = column_headers

# -----------------------
# Scatter Plot b/n 2 vars
# -----------------------

df['engine-size'].replace('?', np.nan, inplace=True)
mean_engine_size = df['engine-size'].astype('float').mean(axis=0)
df['engine-size'].replace(np.nan, mean_engine_size, inplace=True)
df['engine-size'] = df['engine-size'].astype('float')

df['price'].replace('?', np.nan, inplace=True)
mean_price = df['price'].astype('float').mean(axis=0)
df['price'].replace(np.nan, mean_price, inplace=True)
df['price'] = df['price'].astype('float')

x = df['engine-size']
y = df['price']
plt.scatter(x, y)
plt.title('Engine-Size Vs. Price')
plt.xlabel('Engine-Size')
plt.ylabel('Price')

plt.savefig('./scatter.png')
plt.show()

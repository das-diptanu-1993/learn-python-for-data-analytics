import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
df = pd.read_csv(path, header=None)

column_headers = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', '#doors', 'body-style', 
'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
'#cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rmp', 
'city-mpg', 'highway-mpg', 'price']
df.columns = column_headers

#df['drive-wheels'] = df['drive-wheels'].astype('float')
df['price'].replace('?', np.nan, inplace=True)

mean_price = df['price'].astype('float').mean(axis=0)
df['price'].replace(np.nan, mean_price, inplace=True)
df['price'] = df['price'].astype('float')

sns.boxplot(x='drive-wheels', y='price', data=df)

plt.savefig('./box_plot.png')
plt.show()

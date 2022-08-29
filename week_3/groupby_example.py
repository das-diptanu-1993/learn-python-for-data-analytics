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

print(df.dtypes)

df['price'].replace('?', np.nan, inplace=True)
mean_price = df['price'].astype('float').mean(axis=0)
df['price'].replace(np.nan, mean_price, inplace=True)
df['price'] = df['price'].astype('float')

df_test = df[['drive-wheels', 'body-style', 'price']]
print(df_test.head())
df_grp = df_test.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
print(df_grp)
df_pivot = df_grp.pivot(index='drive-wheels', columns='body-style')
print(df_pivot)

fig, ax = plt.subplots()
im = ax.pcolor(df_pivot, cmap='RdBu')

#label names
row_labels = df_pivot.columns.levels[1]
col_labels = df_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(df_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(df_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()
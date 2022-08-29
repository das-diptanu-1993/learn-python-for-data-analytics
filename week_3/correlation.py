import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
df = pd.read_csv(path, header=None)

column_headers = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', '#doors', 'body-style', 
'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
'#cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rmp', 
'city-mpg', 'highway-mpg', 'price']
df.columns = column_headers

# -----------------------------------------
# Pearson Correlation & Correlation Heatmap
# -----------------------------------------

# cleaning up variables: `horsepower`, `price`
df['horsepower'].replace('?', np.nan, inplace=True)
df['horsepower'] = df['horsepower'].astype('float')
df.dropna(subset=['horsepower'], axis=0, inplace=True)

df['price'].replace('?', np.nan, inplace=True)
df['price'] = df['price'].astype('float')
df.dropna(subset=['price'], axis=0, inplace=True)

# calculating `pearson correlation` of 2 variables
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print('Pearson-Correlation: Correlation-Coefficient = {:.2f}; P-Value = {}'.format(pearson_coef, p_value))

# generating heatmap plot b/n different variables
df_vars = df[['horsepower', 'length', 'width', 'height', 'price']]
corr = df_vars.corr()
plt.figure(0)
sns.heatmap(corr, xticklabels=df_vars.columns, yticklabels=df_vars.columns, cmap='Blues')
plt.savefig('./correlation_heatmap.png')
plt.show()
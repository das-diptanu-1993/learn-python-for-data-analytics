import pandas as pd

# import data using file-path and pandas
file_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
df = pd.read_csv(file_path, header=None)
column_heads = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', '#doors', 'body-style', 
'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
'#cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rmp', 
'city-mpg', 'highway-mpg', 'price']
df.columns = column_heads

# column `horsepower` cleanup
import numpy as np
df['horsepower'].replace('?', np.nan, inplace=True)
mean_horsepower = df['horsepower'].astype('float').mean(axis=0)
df['horsepower'].replace(np.nan, mean_horsepower, inplace=True)
df['horsepower'] = df['horsepower'].astype('float')

# normalization of input vector
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df[['horsepower', 'highway-mpg']])
X_scaled = scaler.transform(df[['horsepower', 'highway-mpg']])
print(X_scaled)
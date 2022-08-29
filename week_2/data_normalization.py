import pandas as pd
import numpy as np

# ------------------
# data normalization
# ------------------

# converting .csv (online/offline) into dataframe
file_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
df = pd.read_csv(file_path, header=None)

# setting header for the columns of the dataframe
headers = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', '#doors', 'body-style', 
'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
'#cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rmp', 
'city-mpg', 'highway-mpg', 'price']
df.columns = headers

# data normalization : feature scaling
df['length'] = df['length']/df['length'].max()

# data normalization : min-max
df['width'] = ( df['width'] - df['width'].min() ) / ( df['width'].max() - df['width'].min() )

# data normalization : z-score
df['height'] = ( df['height'] - df['height'].mean() ) / df['height'].std()

# saving dataframe in a .csv file
df.to_csv('./normalized.csv')
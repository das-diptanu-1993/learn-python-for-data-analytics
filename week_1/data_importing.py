import pandas as pd
import numpy as np
from pandasgui import show

# --------------------------
# data importing and reading
# --------------------------

# converting .csv (online/offline) into dataframe
file_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
df = pd.read_csv(file_path, header=None)

# setting header for the columns of the dataframe
headers = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', '#doors', 'body-style', 
'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
'#cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rmp', 
'city-mpg', 'highway-mpg', 'price']
df.columns = headers

# getting first N rows from the dataframe (top/bottom)
top5rows = df.head(5)
print(top5rows)
bottom5rows = df.tail(5)
print(bottom5rows)

# print data type of the columns
print(df.dtypes)

# describe base statistics related to the dataframe
stats = df.describe(include='all').replace(np.nan, '')
print(stats)

# print info related to the dataframe
info = df.info()
print(info)

# saving data in .csv file
df.to_csv('data.csv')

# printing custom unique variants
for data in df.head(5).values:
    print(data)

# iterating through the dataframe
df = df.reset_index()

for index, row in df.iterrows():
    print('Fuel-Type: {}, Body-Style: {}'.format(row['fuel-type'], row['body-style']))

# displaying dataframe using pandas-gui
show(df)




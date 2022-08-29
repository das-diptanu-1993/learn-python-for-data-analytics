import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------
# data binning
# ------------

# converting .csv (online/offline) into dataframe
file_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
df = pd.read_csv(file_path, header=None)

# setting header for the columns of the dataframe
headers = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', '#doors', 'body-style', 
'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
'#cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rmp', 
'city-mpg', 'highway-mpg', 'price']
df.columns = headers

# converting to proper data-type
df['horsepower'].replace('?', np.nan, inplace=True)
avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)
df['horsepower'] = df['horsepower'].astype(int, copy=True)

# visual representation of integer binning
plt.figure(1)
plt.hist(df['horsepower'])
plt.xlabel('horsepower')
plt.ylabel('count')
plt.title('Horsepower Bins')
plt.show()

# creating bins and spliting dataframe by bins
bins = np.linspace(df['horsepower'].min(), df['horsepower'].max(), 4)
group_names = ['LOW', 'MEDIUM', 'HIGH']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True)
print(df['horsepower-binned'].value_counts())

# saving dataframe in a .csv file
df.to_csv('./binned.csv')

# visual representation of binning
plt.figure(2)
plt.hist(df['horsepower'], bins=3)
plt.xlabel('horsepower')
plt.ylabel('count')
plt.title('Horsepower Bins')
plt.show()


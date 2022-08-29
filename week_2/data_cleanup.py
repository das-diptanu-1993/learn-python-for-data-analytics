import pandas as pd
import numpy as np

# --------------------------------
# data cleanup and standardization
# --------------------------------

# converting .csv (online/offline) into dataframe
file_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
df = pd.read_csv(file_path, header=None)

# setting header for the columns of the dataframe
headers = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', '#doors', 'body-style', 
'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
'#cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rmp', 
'city-mpg', 'highway-mpg', 'price']
df.columns = headers

# convert `?` to NumPy.NaN
df.replace('?', np.nan, inplace=True)

# evaluating missing data
missing_data = df.isnull()
proper_data = df.notnull()
print(missing_data)
print(proper_data)

# counting missing values in each column
for col in missing_data.columns.tolist():
    print(missing_data[col].value_counts())

# drop (row/column) data with n/a values
df.dropna(subset=['price'], axis=0, inplace=True)
# df.dropna(axis=1, inplace=True)

# replace missing data with mean
avg_norm_loss = df['normalized-losses'].astype('float').mean(axis=0)
df['normalized-losses'].replace(np.nan, avg_norm_loss, inplace=True)

# replace missing data by most frequent observations
most_frequent = df['#doors'].value_counts().idxmax()
df['#doors'].replace(np.nan, most_frequent)

# convert data-types to proper format
df[['bore', 'stroke']] = df[['bore', 'stroke']].astype('float')
df['normalized-losses'] = df['normalized-losses'].astype('int')

# data standardization : add column
df['city-L/100Km'] = 235/df['city-mpg']

# data standardization : modify column
df['highway-mpg'] = 235/df['highway-mpg']
df.rename(columns={'highway-mpg':'highway-L/100Km'}, inplace=True)

# saving dataframe in a .csv file
df.to_csv('./cleaned_up.csv')




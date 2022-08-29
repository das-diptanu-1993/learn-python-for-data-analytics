import pandas as pd

# ------------------------
# indicator/dummy variable
# ------------------------

# converting .csv (online/offline) into dataframe
file_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
df = pd.read_csv(file_path, header=None)

# setting header for the columns of the dataframe
headers = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', '#doors', 'body-style', 
'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
'#cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rmp', 
'city-mpg', 'highway-mpg', 'price']
df.columns = headers

# checking all the columns in the dataframe
print(df.columns)

# creating indicator/dummy variable to label `fuel-type`
ind_var_1 = pd.get_dummies(df['fuel-type'])
print(ind_var_1.head())

# modifying the indicator/dummy variable columns names
ind_var_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
print(ind_var_1.head())

# merging the indicator/dummy variable columns with dataframe
df = pd.concat([df, ind_var_1], axis=1)

# dropping the `fuel-type` column from the dataframe
df.drop('fuel-type', axis=1, inplace=True)
print(df.head())
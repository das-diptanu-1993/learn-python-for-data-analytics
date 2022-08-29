# -----------
# Import data
# -----------

import pandas as pd
from pandasgui import show 

file_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
df = pd.read_csv(file_path, header=None)
column_heads = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', '#doors', 'body-style', 
'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
'#cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rmp', 
'city-mpg', 'highway-mpg', 'price']
df.columns = column_heads

print('Imported Data:\n{}'.format(df.dtypes))

# ----------------
# Cleaning up data
# ----------------

from numpy import nan, mean

df['price'].replace('?', nan, inplace=True)
df.dropna(subset=['price'], axis=0, inplace=True)
df['price'] = df['price'].astype('float')

df['horsepower'].replace('?', nan, inplace=True)
df['horsepower'] = df['horsepower'].astype('float')
df_hp_mean = df['horsepower'].mean(axis=0)
df['horsepower'].replace(nan, df_hp_mean, inplace=True)

# ----------------------------------------
# Implementing different Regression Models
# ----------------------------------------

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

score = pd.DataFrame(columns=['Model', 'Score'])

# Simple Linear Regression
slr = LinearRegression()
X1 = df[['highway-mpg']]
Y = df['price']
slr.fit(X1, Y)
slr_cv_s_list = cross_val_score(slr, X1, Y, cv=4)
slr_cv_s_mean = mean(slr_cv_s_list)
score = pd.concat( [ score, pd.DataFrame(
        [['SLR', '{:.2f}'.format(slr_cv_s_mean)]], 
        columns=['Model', 'Score'])])

# Multiple Linear Regression
mlr = LinearRegression()
X2 = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
mlr.fit(X2, Y)
mlr_cv_s_list = cross_val_score(mlr, X2, Y, cv=4)
mlr_cv_s_mean = mean(mlr_cv_s_list)
score = pd.concat( [ score, pd.DataFrame(
        [['MLR', '{:.2f}'.format(mlr_cv_s_mean)]], 
        columns=['Model', 'Score'])])

# Multiple Polynomial Regression
pr = PolynomialFeatures(degree=2, include_bias=False)
X2_poly = pr.fit_transform(X2)
mpr = LinearRegression()
mpr.fit(X2_poly, Y)
mpr_cv_s_list = cross_val_score(mpr, X2_poly, Y, cv=4)
mpr_cv_s_mean = mean(mpr_cv_s_list)
score = pd.concat([ score, pd.DataFrame(
    [['MLR', '{:.2f}'.format(mpr_cv_s_mean)]],
    columns=['Model', 'Score']
)])

show(score)







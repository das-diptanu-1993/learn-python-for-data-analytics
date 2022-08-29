# -----------
# Import data
# -----------

from cProfile import label
import pandas as pd

file_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
df = pd.read_csv(file_path, header=None)
column_heads = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', '#doors', 'body-style', 
'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
'#cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rmp', 
'city-mpg', 'highway-mpg', 'price']
df.columns = column_heads

# print('Imported Data:\n{}'.format(df.dtypes))

# ----------------
# Cleaning up data
# ----------------

from numpy import nan, mean

df['price'].replace('?', nan, inplace=True)
df.dropna(subset=['price'], axis=0, inplace=True)
df['price'] = df['price'].astype('float')

df['horsepower'].replace('?', nan, inplace=True)
df['horsepower'] = df['horsepower'].astype('float')
df_horsepower_mean = mean(df['horsepower'])
df['horsepower'].replace(nan, df_horsepower_mean, inplace=True)

# show(df)

# ---------------------
# Model Order Selection
# ---------------------

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

# X = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
X = df[['horsepower']]
Y = df['price']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

Rsqu_test = []
MSE_test = []
order = range(1, 11, 1)

for n in order:
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train)
    x_test_pr = pr.fit_transform(x_test)
    mpr = LinearRegression()
    mpr.fit(x_train_pr, y_train)
    Rsqu_test.append(mpr.score(x_test_pr, y_test))
    y_test_hat = mpr.predict(x_test_pr)
    MSE_test.append(mean_squared_error(y_test, y_test_hat))

Rsqu_norm = []
MSE_norm = []

for i in range(len(order)):
    print('order: {}; score: {:.2f}'.format(order[i], Rsqu_test[i]))
    Rsqu_norm.append( (Rsqu_test[i] - min(Rsqu_test)) / (max(Rsqu_test) - min(Rsqu_test)))
    MSE_norm.append( (MSE_test[i] - min(MSE_test)) / (max(MSE_test) - min(MSE_test)) )

plt.figure()
plt.plot(order, Rsqu_norm, label='R^2')
plt.plot(order, MSE_norm, label='MSE')
plt.xlabel('order')
plt.ylabel('value')
plt.legend()
plt.show()

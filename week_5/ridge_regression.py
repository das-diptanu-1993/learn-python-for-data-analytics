import warnings
warnings.filterwarnings('ignore')

# -----------
# Import data
# -----------

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

from numpy import nan, mean, meshgrid, array

df['price'].replace('?', nan, inplace=True)
df.dropna(subset=['price'], axis=0, inplace=True)
df['price'] = df['price'].astype('float')

df['horsepower'].replace('?', nan, inplace=True)
df['horsepower'] = df['horsepower'].astype('float')
df_horsepower_mean = mean(df['horsepower'])
df['horsepower'].replace(nan, df_horsepower_mean, inplace=True)

# ----------------
# Ridge Regression
# ----------------

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from pandasgui import show
# from matplotlib import pyplot as plt

X = df[['horsepower']]
Y = df['price']

order = range(1, 11, 1)
alpha = [0, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
cv_score = pd.DataFrame(columns=order, index=alpha)

for l in alpha:
    cv_n_score = []
    for n in order:
        process_list = [
            ('scale', StandardScaler(with_mean=False)), 
            ('poly', PolynomialFeatures(degree=n, include_bias=False)), 
            ('model', Ridge(alpha=l)
        )]
        pipe = Pipeline(process_list)
        pipe.fit(X, Y)
        pipe_cv_score_list = cross_val_score(pipe, X, Y, cv=4)
        pipe_cv_score = mean(pipe_cv_score_list)
        print('order: {:.2f}, alpha: {:.2f}, score: {:.2f}'.format(n, l, pipe_cv_score))
        cv_score.loc[l, n] = pipe_cv_score

show(cv_score)



'''
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

O, A = meshgrid(order, alpha)
CV = array(cv_score)
ax.plot_surface(O, A, CV, alpha=0.5, rstride=1, cstride=1)
ax.set_title('Cross Validation Score of Ridge Regression')
ax.set_xlabel('order')
ax.set_ylabel('alpha')
# ax.set_yscale('log')
ax.set_zlabel('score')
plt.show()

'''







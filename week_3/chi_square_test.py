import pandas as pd
from scipy import stats

path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
df = pd.read_csv(path, header=None)

column_heads = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', '#doors', 'body-style', 
'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
'#cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rmp', 
'city-mpg', 'highway-mpg', 'price']
df.columns = column_heads

print(df['fuel-type'].value_counts())
print(df['aspiration'].value_counts())

cross_tab = pd.crosstab(df['fuel-type'], df['aspiration'], rownames=['fuel-type'], colnames=['aspiration'])
print('\nCross-Tab: \n{}'.format(cross_tab))

analysis = stats.chi2_contingency(cross_tab)
(chi_square_value, p_value, degree_of_freedom, expected_cross_tab) = analysis
print('\nChi-Square Association Test b/n `fuel-type`, `aspiration`:\nChi-Square-Value: \t{}\nP-Value: \t\t{}\nDegree-of-Freedom: \t{}\nExpected-Cross-Tab: \n{}'
        .format(chi_square_value, p_value, degree_of_freedom, expected_cross_tab))


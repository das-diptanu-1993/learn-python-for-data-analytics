import pandas as pd
import numpy as np
from pandasgui import show

path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
df = pd.read_csv(path, header=None)

column_headers = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', '#doors', 'body-style', 
'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
'#cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rmp', 
'city-mpg', 'highway-mpg', 'price']
df.columns = column_headers

df['price'].replace('?', np.nan, inplace=True)
df.dropna(subset=['price'], axis=0, inplace=True)
df['price'] = df['price'].astype('float')

df_temp = df[['drive-wheels', 'body-style', 'price']]
df_grp = df_temp.groupby(['drive-wheels', 'body-style'], as_index=False).mean()

df_pivot = df_grp.pivot(index='drive-wheels', columns='body-style')

df_body_style = df['body-style'].value_counts().to_frame()
df_body_style.rename(columns={'index':'body-style_name', 'body-style':'body-style_count'}, inplace=True)

df_grp1 = df_temp.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
df_grp1.rename(columns={'price':'price-mean'}, inplace=True)
df_grp1['identifier'] = df_grp1['drive-wheels'] + df_grp1['body-style']

df_grp2 = df_temp.groupby(['drive-wheels', 'body-style'], as_index=False).median()
df_grp2.rename(columns={'price':'price-median'}, inplace=True)
df_grp2['identifier'] = df_grp2['drive-wheels'] + df_grp2['body-style']
df_grp2.drop(['drive-wheels', 'body-style'], axis=1, inplace=True)

df_grp3 = df_temp.groupby(['drive-wheels', 'body-style'], as_index=False).std()
df_grp3.rename(columns={'price':'price-std_deviation'}, inplace=True)
df_grp3['identifier'] = df_grp3['drive-wheels'] + df_grp3['body-style']
df_grp3.drop(['drive-wheels', 'body-style'], axis=1, inplace=True)

df_group_stats = pd.merge(df_grp1, df_grp2, how='inner', on='identifier')
df_group_stats = pd.merge(df_group_stats, df_grp3, how='inner', on='identifier')
df_group_stats.drop(['identifier'], axis=1, inplace=True)

df_agg = df.groupby(['drive-wheels', 'body-style']).agg({'price':['mean','median','max',lambda x : np.percentile(x, q = 50)]})
df_agg.rename(columns={'<lambda_0>':'percentile-50'}, inplace=True)

show(df_grp, df_pivot, df_body_style, df_group_stats, df_agg)
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# pandas for data input, output, and manipulation
# using series and datasets
import pandas as pd

df = pd.read_csv('Datasets/direct_marketing.csv')
print(df.describe())
df.summary
print(df.columns)
print(df.recency)
#using booleans
print(df[df.channel == "Phone"])
#df['recency'] returns series, df[['recency']] dataset
#likewise, df.loc[:, 'recency'] vs [:, ['recency]]
#use iloc for index based retrieval
df = pd.get_dummies(df, columns=['zip_code'])
#explodes one column into multiple for binary numeric data
"""
Can also use ordered_list and 
df.feature = df.feature.astype("category", ordered=True,
categories=ordered_list).cat.codes
to create integer based column for nominal String features

or just
df['feature'] = df.feature.astype("category").cat.codes for
non-nominal
"""
#can use sklearn.feature_extraction.text CountVectorizer to 
#make bodies of text features!

print(df.head())

df.to_excel('directMarketing.xlsx')

"""
WRANGLING
df.my_feature.fillna(df.my_feature.mean())
df.fillna(0)
df.fillna(method='ffill') or bfill
df.fillna(limit = 5)
df.interpolate(method='polynomial', order=2)

df.dropna(axis=0) //rows! 
df = df.drop(labels=['features'], axis=1)
//can also to in-place to be more space-efficient
//via inplace=True
df = df.drop_duplicates(subset=['features'])
df = df.reset_index(drop=True)

df.dtypes
.to_datetime(), .to_numeric()
e.g. df.Date = pd.to_datetime(df.Date, errors='coerce')

view unique values: df.feature.unique()
view counts: df.feature.value_counts()
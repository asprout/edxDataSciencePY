# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 21:01:34 2017

@author: Ling
"""
#CLUSTERING
#1. For assigning data to clusters (e.g. recommendations)
#2. For finding optimal centroids (e.g. school locations)
#again, assumes length normalization
"""
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
kmeans.fit(df)
labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_
"""
"""
For machine learning in python, we've had to use .fit()
for each class.
For unsupervisted methods:
    .transform() alters values of existing features
    .predict() predicts labels of specified samples with clustering
For supervised methods:
    .predict() predicts labels of new samples
    .predict_proba() gives prob of sample label
    .score() scores how well your model fits training data
    
Transformation and modelling should only be done on training
data - data should be split to have enough data to train
AND test your model!
Avoid overfitting data!
"""
#SPLITTING
#from sklearn.model_selection import train_test_split
""" 
E.G.
>>> data   = [0,1,2,3,4, 5,6,7,8,9]  # input dataframe samples
>>> labels = [0,0,0,0,0, 1,1,1,1,1]  # the function we're training is " >4 "

>>> data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.5, random_state=7)
//passing in random_state helps make centroid selection reproducible
>>> data_train
[9, 7, 3, 6, 4]

>>> label_train
[1, 1, 0, 1, 0]

>>> data_test
[8, 5, 0, 2, 1]

>>> label_test
[1, 1, 0, 0, 0]

from sklearn.metrics import accuracy_score
predictions = my_model.predict(data_test)
accuracy_score(label_test, predictions)
"""
#CLASSIFICATION
#K-Nearest Neighbors
#data needs to be measurable... 
#make sure to take out classifiers
"""
>>> X_train = pd.DataFrame([ [0], [1], [2], [3] ])
>>> y_train = [0, 0, 1, 1]

>>> from sklearn.neighbors import KNeighborsClassifier
>>> model = KNeighborsClassifier(n_neighbors=3)
>>> model.fit(X_train, y_train) 

>>> # You can pass in a dframe or an ndarray
>>> model.predict([[1.1]])

>>> model.predict_proba([[0.9]])
[[ 0.66666667  0.33333333]]
"""

#LINEAR REGRESSION
#using ordinary least squares
#ASSUMES LINEAR INDEPENDENCE OF FEATURES!
"""
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

intercept_ : the scalar constant offset value
coef_ : an array of weights, one per input feature
.score() returns R2 coefficient 
"""
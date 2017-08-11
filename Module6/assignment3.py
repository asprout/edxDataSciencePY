import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing

from sklearn.decomposition import PCA
from sklearn import manifold

X = pd.read_csv("Datasets/parkinsons.data", index_col = 0)
y = X['status']
X = X.drop(labels=['status'], axis=1)

X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=0.3, random_state=7)

print(X_train.head())

scale = preprocessing.StandardScaler()
#normalizer, maxabs, minmax, kernelcenterer, standard
scale.fit(X_train)
X_train = pd.DataFrame(scale.transform(X_train))
X_test = pd.DataFrame(scale.transform(X_test))

#scaled = pd.DataFrame(scaled, columns = df.columns)

print(X_train.head())
best_score = 0
for a in range(4, 15, 1):
#if True:
    pca = PCA(n_components = a, svd_solver='full')
    pca.fit(X_train)
    Xtrain = pd.DataFrame(pca.transform(X_train))
    Xtest = pd.DataFrame(pca.transform(X_test))
    model = SVC(C = 1.55, gamma = 0.097)
    model.fit(Xtrain, y_train)
    sc = model.score(Xtest, y_test)
    if (sc > best_score):
        best_score = sc


for n in range(2, 6, 1):
    for c in range(4, 7, 1):
        iso = manifold.Isomap(n_neighbors = n, n_components = c)
        iso.fit(X_train)
        X_train = iso.transform(X_train)
        X_test = iso.transform(X_test)
        model = SVC(C = 1.55, gamma = 0.097)
        model.fit(X_train, y_train)
        sc = model.score(X_test, y_test)
        if (sc > best_score):
            best_score = sc

print("Best Score: " + str(best_score))
          
model = SVC(C = 1.55, gamma = 0.097)
model.fit(X_train, y_train)

print("First Score: " + str(model.score(X_test, y_test)))

best_c = 0.05
best_g = 0.001

for c in np.arange(0.05, 2, 0.05):
    for g in np.arange(0.001, 0.1, 0.001):
        model = SVC(C = c, gamma = g)
        model.fit(X_train, y_train)
        cur_score = model.score(X_test, y_test)
        if (cur_score > best_score):
            best_score = cur_score
            best_c = c
            best_g = g
print("Best Parameters Score: " + str(best_score))
print("with c and g as: " + str(best_c) + ", " + str(best_g))

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:42:34 2017

@author: Ling
"""

#PRINCIPAL COMPONENT ANALYSIS
#unsupervised learning algorithm
#attempts to convert features into set of linearly uncorrelated ones
#aka dimensional reduction
#ordered by importance 
#the more variance expressed in a feature, the more important it is
#MAKE SURE TO NORMALIZE or inherent variance differences!
#e.g. using StandardScaler
#from sklearn import preprocessing
#scaled = preprocessing.StandardScaler().fit_transform(df)
#scaled = pd.DataFrame(scaled, columns = df.columns)
#randomizedPCA is faster for larger datasets, sacrificing accuracy
"""
all dimensionality reduction methods, have three main uses:

reducing the dimensionality and thus complexity of your dataset.
preparation for other supervised learning tasks
To make visualizing your data easier
"""
"""
from sklearn.decomposition import PCA
pca = PCA(n_components = 2, svd_solver='full')
pca.fit(df)
//has components_, explained_variance_, explained_variance_ratio_
T = pca.transform(df)
"""

#ISOMAP: when non-linear relationships exist
"""
from sklearn import manifold
iso = manifold.Isomap(n_neighbors=4, n_components=2)
iso.fit(df)
Isomap(eigen_solver='auto', max_iter=None, n_components=2, n_neighbors=4,
    neighbors_algorithm='auto', path_method='auto', tol=0)
manifold = iso.transform(df)
"""
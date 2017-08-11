# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 18:58:50 2017

@author: Ling
"""
#SVM and SVC: Support vector machines/classifiers
#support vectors - objects from either class closest to one another
#algorithm maximizes margin between data
#the kernel trick
#allow linear separation of data to compute how similar 
#two samples are
#SVC: when classification speed is more critical than training speed
"""
from sklearn.svm import SVC
model = SVC(kernel='linear')
    C: penalty parameter, 
    gamma inversely proportional to influence of training samples
model.fit(X, y)
.decision_function(X) calculates distance of a set of samples
to the deciison boundary
"""
#DECISION TREES
#supervised, probabilistic classifier
#using entropy, or information gain
#indifferent to feature scaling!

"""
Unlike SVMs, the accuracy of a DTree doesn't decrease when you include irrelevant features
Unlike KNeighbors, both training and predicting with a DTree are relatively fast operations
Unlike PCA / IsoMap, DTrees are invariant to monotonic feature scaling and transformations
Moreover, a trained DTree model is readily human inspectable
from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=9, criterion='entropy')
#default Gini, info gain
#splitter, max_features
#feature_importances
model.fit(X, y)

if brew install graphviz:
    tree.export_graphviz(model.tree_, out_file='tree.dot', feature_names = X.columns)
    from subprocess import call
    call(['dot', '-T', 'png', 'tree.dot', '-o', 'tree.png'])
"""

#RANDOM FORESTS
#Slower but more accurate than DTrees... generally
#Using the mode of overfitted decision trees
#(each tree randomly samples data from training data)
#bootstrapping, or randomization of samples of each tree, makes sure
#the trees are not correlated
#on the tree level, each tree also 'feature-bags'
#there is an out-of-bag error metric...
#but separate data into train+test sets anyways (obb_score = False)
"""
from sklearn.ensemble import RandomForestClasifier
model = RandomForestClassifier(n_estimators=10, oob_score=True)
model.fit(X, y)
model.oob_score_
model.estimators_ : structure of individual decision trees
"""
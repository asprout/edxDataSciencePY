# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 19:38:55 2017

@author: Ling
#copied from module 7 notes
"""

#MOD7
"""
>>> import matplotlib.pyplot as plt

>>> columns = ['Cat', 'Dog', 'Monkey']
>>> confusion = metrics.confusion_matrix(y_true, y_pred)

>>> plt.imshow(confusion, cmap=plt.cm.Blues, interpolation='nearest')
>>> plt.xticks([0,1,2], columns, rotation='vertical')
>>> plt.yticks([0,1,2], columns)
>>> plt.colorbar()

>>> plt.show()
"""
"""
>> import sklearn.metrics as metrics
>>> y_true = [1, 1, 2, 2, 3, 3]  # Actual, observed testing datset values
>>> y_pred = [1, 1, 1, 3, 2, 3]  # Predicted values from your model

>>> metrics.accuracy_score(y_true, y_pred)
0.5

>>> metrics.accuracy_score(y_true, y_pred, normalize=False)
3

>>> metrics.recall_score(y_true, y_pred, average='weighted')
0.5

>>> metrics.recall_score(y_true, y_pred, average=None)
array([ 1. ,  0. ,  0.5])

Precision
You can also calculated the precision score. It is defined very similarly: true_positives / (true_positives + false_positives). The only difference is the very last term in the equation:

>>> metrics.precision_score(y_true, y_pred, average='weighted')
0.38888888888888884

>>> metrics.precision_score(y_true, y_pred, average=None)
array([ 0.66666667,  0.        ,  0.5       ])

F1
The F1 Score is a weighted average of the precision and recall. Defined as 2 * (precision * recall) / (precision + recall), the best possible result is 1 and the worst possible score is 0:

>>> metrics.f1_score(y_true, y_pred, average='weighted')
0.43333333333333335

>>> metrics.f1_score(y_true, y_pred, average=None)
array([ 0.8,  0. ,  0.5])

>>> target_names = ['Fruit 1', 'Fruit 2', 'Fruit 3']
>>> metrics.classification_report(y_true, y_pred, target_names=target_names)

"""
"""
>>> from sklearn.cross_validation import train_test_split
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

>>> # Test how well your model can recall its training data:
>>> model.fit(X_train, y_train).score(X_train, y_train)
0.943262278808

>>> # Test how well your model can predict unseen data:
>>> model.fit(X_test, y_test).score(X_test, y_test)
#is this a typo?... who knows
0.894716422024

cross_val_score().

This method takes as input your model along with your training dataset and performs K-fold cross validations on it. In other words, your training data is first cut into a number of "K" sets. Then, "K" versions of your model are trained, each using an independent K-1 number of the "K" available sets. Each model is evaluated with the last set, it's out-of-bag set. If this sounds super familiar to you, it's because this is the same bootstrapping technique used in random forest.

# 10-Fold Cross Validation on your training data
>>> from sklearn import cross_validation as cval
sklearn.model_selection.cross_val_score
>>> cval.cross_val_score(model, X_train, y_train, cv=10)
array([ 0.93513514,  0.99453552,  0.97237569,  0.98888889,  0.96089385,
        0.98882682,  0.99441341,  0.98876404,  0.97175141,  0.96590909])

>>> cval.cross_val_score(model, X_train, y_train, cv=10).mean()
0.97614938602520218
"""
"""
In the wild, the best process to use depending on how many samples you have at your disposal and the machine learning algorithms you are using, is either of the following:

Split your data into training, validation, and testing sets.
Setup a pipeline, and fit it with your training set
Access the accuracy of its output using your validation set
Fine tune this accuracy by adjusting the hyperparamters of your pipeline
when you're comfortable with its accuracy, finally evaluate your pipeline with the testing set
- OR - 

Split your data into training and testing sets.
Setup a pipeline with CV and fit / score it with your training set
Fine tune this accuracy by adjusting the hyperparamters of your pipeline
When you're comfortable with its accuracy, finally evaluate your pipeline with the testing set
"""
"""
GridSearchCV takes care of your parameter tuning and also tacks on end-to-end cross validation.
In its simplest form, GridSearchCV works by taking in an estimator, a grid of parameters you want optimized, and your cv split value. This is the example from SciKit-Learn's API page:

>>> from sklearn import svm, grid_search, datasets

>>> iris = datasets.load_iris()
>>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 5, 10]}
>>> model = svm.SVC()

>>> classifier = grid_search.GridSearchCV(model, parameters)
>>> classifier.fit(iris.data, iris.target)
GridSearchCV(cv=None, error_score='raise',
       estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False),
       fit_params={}, iid=True, n_jobs=1,
       param_grid={'kernel': ('linear', 'rbf'), 'C': [1, 5, 10]},
       pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=0)
In addition to explicitly defining the parameters you want tested, you can also use randomized parameter optimization with SciKit-Learn's RandomizedSearchCV class.
>>> parameter_dist = {
  'C': scipy.stats.expon(scale=100),
  'kernel': ['linear'],
  'gamma': scipy.stats.expon(scale=.1),
}

>>> classifier = grid_search.RandomizedSearchCV(model, parameter_dist)
>>> classifier.fit(iris.data, iris.target)
"""
"""
 SciKit-Learn has created a pipelining class. It wraps around your entire data analysis pipeline from start to finish, and allows you to interact with the pipeline as if it were a single white-box, configurable estimator. The other added benefit is that once your pipeline has been built, since the pipeline inherits from the estimator base class, you can use it pretty much anywhere you'd use regular estimators—including in your cross validator method. Doing so, you can simultaneously fine tune the parameters of each of the estimators and predictors that comprise your data-analysis pipeline.
Every intermediary model, or step within the pipeline must be a transformer.
The very last step in your analysis pipeline only needs to implement the .fit() method, since it will not be feeding data into another step
>>> from sklearn.pipeline import Pipeline

>>> svc = svm.SVC(kernel='linear')
>>> pca = RandomizedPCA()

>>> pipeline = Pipeline([
  ('pca', pca),
  ('svc', svc)
])
>>> pipeline.set_params(pca__n_components=5, svc__C=1, svc__gamma=0.0001)
>>> pipeline.fit(X, y)
A very nifty hack you should be aware of to circumvent this is by writing your own transformer class, which simply wraps a predictor and masks it as a transformer:

from sklearn.base import TransformerMixin

class ModelTransformer(TransformerMixin):
  def __init__(self, model):
    self.model = model

  def fit(self, *args, **kwargs):
    self.model.fit(*args, **kwargs)
    return self

  def transform(self, X, **transform_params):
    # This is the magic =)
    return DataFrame(self.model.predict(X))
"""
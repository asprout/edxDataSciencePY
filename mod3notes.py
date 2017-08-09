# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 12:47:19 2017
VISUALIZATIONS
@author: Ling
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('ggplot')

student_dataset = pd.read_csv("Datasets/students.data", index_col = 0)

my_series = student_dataset.G3
my_dataframe = student_dataset[['G3', 'G2', 'G1']]

my_series.plot.hist(alpha=0.5)
my_dataframe.plot.hist(alpha=0.5)
#normed = True can normalize results as percentages

student_dataset.plot.scatter(x='G1', y='G3')

#3-D plots
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Final Grade')
ax.set_ylabel('First Grade')
ax.set_zlabel('Daily Alcohol')
ax.scatter(student_dataset.G1, student_dataset.G3, student_dataset['Dalc'], c='b', marker='o')

plt.show()

#multi-dimensional: parallel coordinate plots
from sklearn.datasets import load_iris
from pandas.plotting import parallel_coordinates

data = load_iris()
df = pd.DataFrame(data.data, columns = data.feature_names)
df['target_names'] = [data.target_names[i] for i in data.target]

plt.figure()
parallel_coordinates(df, 'target_names')
plt.show()
#normalize if necessary because only one scale for axis

#an Andrew's plot visualizes multivariate data
#by plotting observations as a curve

from pandas.plotting import andrews_curves
plt.figure()
andrews_curves(df, 'target_names')
plt.show()

#use .imshow() for correlation
import numpy as np
df = pd.DataFrame(np.random.randn(1000, 5), columns = ['a', 'b', 'c', 'd', 'e'])
df.corr()
plt.imshow(df.corr(), cmap = plt.cm.Blues, interpolation='nearest')
plt.colorbar()
plt.show()


# -*- coding: utf-8 -*-
"""MLDecisionTree.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NRSL6twGx_pgEr3iO_jN4UNiMY7yZSBZ
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('/content/PlayTennis.csv')
data.head()

X = data.copy()
y = X.pop('Play Tennis')

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

arr = X.columns

"""You have to do One-Hot or OrdinalEncoding as Desicion Tree only works if it can convert data into float32"""

from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
X = oe.fit_transform(X)

model = DecisionTreeClassifier()
model.fit(X, y)
plt.figure(figsize = (10, 8))
tree.plot_tree(model)

# pip install graphviz

import graphviz

output = tree.export_graphviz(model, out_file=None, feature_names=arr, class_names=y.unique(), filled=True, rounded = True, special_characters=True)

graph = graphviz.Source(output)
graph

from sklearn.ensemble  import RandomForestClassifier
from sklearn.tree import export_graphviz
forest = RandomForestClassifier(n_estimators=10)

forest.fit(X, y)

for i in range(len(forest.estimators_)):
  plt.figure(figsize = (10, 8))
  tree.plot_tree(forest.estimators_[i])

"""All the trees are plotted here but you have to scroll down to see how they are, don't know how this will look in spyder or jupyter notebook"""

forest.predict([[0.4, .2, .2, .1]])
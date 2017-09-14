#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 12:45:35 2017

@author: lucassantos
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

def processData (df):
    mapping = { "female": 0, "male": 1, "C": 0, "S": 1, "Q": 2 }
    y = df['Survived']
    X = df.drop(dstr.columns[[0, 1, 3, 8, 10]], axis=1).applymap(lambda s: mapping.get(s) if s in mapping else s).apply(pd.to_numeric)
    X = X.drop(X.columns[[2]], axis=1)
    
    return X, y


dstr = pd.read_csv('train.csv', keep_default_na=False)
dsts = pd.read_csv('test.csv', keep_default_na=False)

train, test = train_test_split(dstr, test_size = 0.2)

X_train, y_train = processData (train)
X_test, y_test = processData (test)

# Modelos para experimentar SVM, K-Nearest Neighbors, Logistic Regression, Perceptron
models = [ svm.SVC() ]
model = models[0]

print (model)
print (X_train)
print (y_train)

np.isnan(X_train)

# for model in models:
model.fit (X_train, y_train)
y_pred = model.predict (X_test)
report = classification_report(y_test, y_pred, target_names=target_names)
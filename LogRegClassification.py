# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 23:46:55 2020

@author: chani
"""
#binary & mutli calssification problem of logisticRegression
import sklearn as sk
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os

bankdata = pd.read_csv("E:/Data science/Python/bill_authentication.csv")
X = bankdata.drop('Class', axis=1)
y = bankdata['Class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X_train, y_train)
#LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_tr, y_tr) for multiclassification
'''LR.predict(X.iloc[460:,:])round(LR.score(X,y), 4)''''
y_pred = LR.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
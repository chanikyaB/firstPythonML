# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:55:12 2020

@author: chani
"""

#decision tree classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
dataset = pd.read_csv("E:/Data science/Python/bill_authentication.csv")
dataset.shape
dataset.head()
#preparing the data
X = dataset.drop('Class', axis=1)
y = dataset['Class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
#training n making predictions
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#evaluating alogorithm
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#accuracy score
from sklearn.metrics import accuracy_score
print( accuracy_score(y_test, y_pred))
#decision tree regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
dataset = pd.read_csv('E:/Data science/Python/petrol_consumption.csv')
dataset.head()
dataset.describe()
#preparing the data
X = dataset.drop('Petrol_Consumption', axis=1)
y = dataset['Petrol_Consumption']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#training n making predictions
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
df
#evaluating the algorithm: To evaluate performance of the regression algorithm, the commonly used metrics are mean absolute error, mean squared error, and root mean squared error
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#The mean absolute error for our algorithm is 55, which is less than 10 percent of the mean of all the values in the 'Petrol_Consumption' column. mean of petrol conusmption is 576.This means that our algorithm did a fine prediction job.
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 19:41:52 2020

@author: chani
"""

#MLR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('E:/Data science/Python/petrol_consumption.csv')
dataset.head()
dataset.describe()
#preparing data
X = dataset[['Petrol_tax', 'Average_income', 'Paved_Highways',
       'Population_Driver_licence(%)']]
y = dataset['Petrol_Consumption']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
coeff_df
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#You can see that the value of root mean squared error is 60.07, which is  lesser  than 10% of the mean value of the gas consumption
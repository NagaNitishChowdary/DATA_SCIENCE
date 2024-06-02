# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 09:48:33 2024

@author: Naga Nitish
"""

# SIMPLE LINEAR REGRESSION 

import numpy as np 

# we need to give x as columns 
age = np.array([[20],[24],[28],[32],[36],[40]])
weight = np.array([70,66,78,72,86,90])

import matplotlib.pyplot as plt 
plt.scatter(age,weight)
plt.show()


# fit the model 
from sklearn.linear_model import LinearRegression
LR = LinearRegression()

LR.fit(age,weight)

# B0 value ---> intercept 
LR.intercept_   # 44.0 

# B1 value ---> slope
LR.coef_   # array([1.1])



Y_actual = weight 
Y_pred = LR.predict(age)

Y_actual # array([70, 66, 78, 72, 86, 90])
Y_pred # array([66. , 70.4, 74.8, 79.2, 83.6, 88. ])


import matplotlib.pyplot as plt 
plt.scatter(age,Y_actual)
plt.scatter(age,Y_pred,color='red')
plt.show()


# =================================================

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_actual,Y_pred)

print("Mean Squared Error(mse) ",mse.round(4))
print("Root Mean Squared Error(RMSE) ",np.sqrt(mse).round(4))

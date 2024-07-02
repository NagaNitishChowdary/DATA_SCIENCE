# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 23:26:01 2024

@author: Naga Nitish
"""

# ====================================================

# step 1 ---> collecting the data 

import pandas as pd 
import numpy as np 

df = pd.read_csv('2_Cars_4vars.csv')
df

# ===================================================

# step 2 ---> doing Exploratory Data Analysis (EDA) such as 
# scatter plot,histogram, boxplot, ....

# ===> step 2 is to understand more about the data 

import matplotlib.pyplot as plt 

plt.scatter(df['HP'],df['MPG'])
plt.show()

plt.scatter(df['SP'],df['MPG'])
plt.show()

plt.scatter(df['VOL'],df['MPG'])
plt.show()

plt.scatter(df['WT'],df['MPG'])
plt.show()


# Finding correlation between the variables so that we can prioritize 
# the first for fitting the model.
print(df.corr())

#          HP       MPG       VOL        SP        WT
# HP   1.000000 -0.725038  0.077459  0.973848  0.076513
# MPG -0.725038  1.000000 -0.529057 -0.687125 -0.526759
# VOL  0.077459 -0.529057  1.000000  0.102170  0.999203
# SP   0.973848 -0.687125  0.102170  1.000000  0.102439
# WT   0.076513 -0.526759  0.999203  0.102439  1.000000

# MPG is target variable 
# with MPG, HP has highest correlation 
# then after, SP has second highest correlation .........

# ==================================================== 

# step 3 ---> data cleaning
# step 4 ---> data transformation 
# step 5 ---> data partition

# ====================================================

Y = df['MPG']

# X = df[['HP']]  # MODEL 1 
# X = df[['HP','SP']]  # MODEL 2 
# X = df[['HP','SP','VOL']]  # MODEL 3 
# X = df[['HP','SP','VOL','WT']]  # MODEL 4 



#  MODEL                   MSE     RMSE    R2_SCORE  REMARKS  

#   HP                    39.06    6.25      53
#   HP,SP                 38.49    6.2       53      NO IMPROVEMENT
#   HP,SP,VOL             18.91    4.35      77      PERFORMANCE IMPROVEMENT
#   HP,SP,VOL,WT          18.9     4.35      77      NO IMPROVEMENT


# ADDING THE SP & WT ---> NO CHANGE IN R2 ---> NEGLIGIBLE CONTRIBUTION
# THERE ARE NO USAGE FOR MODEL DEVELOPMENT ---> SO WE CAN REMOVE IT 


# THOSE X VARIABLES CONTRIBUTION IS NOT MORE USAGE FOR MODEL DEVELOPMENT

# IN THE GIVEN X VARIABLES, THERE IS PROBLEM CALLED "MULTICOLLINEARITY" ISSUES
# MULTICOLLINEARITY ISSUES WILL EXISTS IN BETWEEN THE X VARIABLES 

# WHEN 2 X VARIABLES ARE HIGHLY CORRELATED IN BETWEEN THEM RATHER THEM WITH 
# Y, THEN IT LEADS TO MULTICOLLINEARITY ISSUES. 

# AS HP & SP HAS CORRELATION OF 0.97 ---> SO ADDING BOTH OF THEM WILL NOT 
# IMPROVE PERFORMANCE ---> SAME TO  VOL & WT (0.99)


# TAKE ONLY X VARIABLE AS PER OUR CHOICE OUT OF TWO FOR MODEL FITTING 

X = df[['HP','VOL']]


# step 6 ---> fit the model 


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)

# B0 value ---> intercept 
LR.intercept_   # 44.0 

# B1 value ---> slope
LR.coef_   # array([1.1])

Y_pred = LR.predict(X)


# ===========================================

# step 8 ---> Evaluating the model using metrics 

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y,Y_pred)

print("Mean Squared Error(mse) ",mse.round(4))  # 17.8667
print("Root Mean Squared Error(RMSE) ",np.sqrt(mse).round(4))  # 4.2269

# Accuracy percentage of model ---> r2 score 
r2 = r2_score(Y,Y_pred).round(2)
print("R2 score ",r2)   # 0.76

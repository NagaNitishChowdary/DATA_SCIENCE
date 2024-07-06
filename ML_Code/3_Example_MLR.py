# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 01:14:00 2024

@author: Naga Nitish
"""

# ===============================================

import numpy as np 
import pandas as pd 

# step 1 ---> collecting the data 

df = pd.read_csv('2_bloodpress.csv')
df

# ===============================================

# step 2 ---> doing EDA such as scatter plot,box plot,histogram ....

# ===> step 2 is to understand more about the data 

import matplotlib.pyplot as plt 

plt.scatter(df['BP'],df['Age'])
plt.show()

plt.scatter(df['BP'],df['Weight'])
plt.show()

plt.scatter(df['BP'],df['BSA'])
plt.show()

plt.scatter(df['BP'],df['Dur'])

plt.scatter(df['BP'],df['Pulse'])

plt.scatter(df['BP'],df['Stress'])
plt.show()


# Finding correlation between the variables so that we can prioritize 
# the first for fitting the model.
df.corr()

#	       Pt	       BP	       Age	     Weight	       BSA	      Dur	     Pulse	     Stress
# Pt	  1.000000	0.031135	0.042694	0.024857   -0.031288	0.176246	0.112285	0.343152
# BP	  0.031135	1.000000	0.659093	0.950068	0.865879	0.292834	0.721413	0.163901
# Age	  0.042694	0.659093	1.000000	0.407349	0.378455	0.343792	0.618764	0.368224
# Weight  0.024857	0.950068	0.407349	1.000000	0.875305	0.200650	0.659340	0.034355
# BSA	 -0.031288	0.865879	0.378455	0.875305	1.000000	0.130540	0.464819	0.018446
# Dur	  0.176246	0.292834	0.343792	0.200650	0.130540	1.000000	0.401514	0.311640
# Pulse	  0.112285	0.721413	0.618764	0.659340	0.464819	0.401514	1.000000	0.506310
# Stress  0.343152	0.163901	0.368224	0.034355	0.018446	0.311640	0.506310	1.000000

# BP is target variable 
# with BP , WEIGHT has highest correlation 
# then after, BSA has second highest correlation .........

# ======================================================

# step 3 ---> data cleaning 
# step 4 ---> data transformation 
# step 5 ---> data partition

# ======================================================

Y = df['BP']

# X = df[['Weight']]  # MODEL 1 
# X = df[['Weight','BSA']]  # MODEL 2
# X = df[['Weight','BSA','Pulse']]  # MODEL 3
# X = df[['Weight','BSA','Pulse','Age']]  # MODEL 4
# X = df[['Weight','BSA','Pulse','Age','Dur']]  # MODEL 5
# X = df[['Weight','BSA','Pulse','Age','Dur','Stress']]  # MODEL 6


#  MODEL                               MSE     RMSE    R2_SCORE  REMARKS  

#   Weight                            2.726    1.65      90
#   Weight,BSA                        2.585    1.60      91      SLIGHT IMPROVEMENT
#   Weight,BSA,Pulse                  1.905    1.38      93      PERFORMANCE IMPROVEMENT
#   Weight,BSA,Pulse,Age              0.150    0.38      99      IMPROVEMENT
#   Weight,BSA,Pulse,Age,Dur          0.129    0.36      100     NO IMPROVEMENT
#   Weight,BSA,Pulse,Age,Dur,Stress   0.10     0.32      100

# Adding Stress doesn't makes sense as it is already giving maximum r2_score

# step 6 ---> fit the model 


X = df[['Weight','BSA','Pulse','Age','Dur']]

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

print("Mean Squared Error(mse) ",mse.round(4))
print("Root Mean Squared Error(RMSE) ",np.sqrt(mse).round(4))  

# Accuracy percentage of model ---> r2 score 
r2 = r2_score(Y,Y_pred).round(2)
print("R2 score ",r2)   

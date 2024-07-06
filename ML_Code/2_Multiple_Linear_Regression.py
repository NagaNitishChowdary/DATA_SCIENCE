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




# ======================================================

# To calculate r square to find VIF value (varience influence factor)

# VIF < 5  ---> MODEL IS EXCELLENT I.E. NO MULTICOLLINEARITY ISSUES 
#     < 10 ---> MODEL IS HAVING SLIGHTLY MULTICOLLINEARITY ISSUES , BUT STILL ACCEPTABLE
#     > 10 ---> MODEL CANNOT BE ACCEPTABLE 


import statsmodels.formula.api as smf 
model = smf.ols("Y ~ X",data = df).fit() 
# model = smf.ols("HP ~ VOL",data = df).fit()
r2 = model.rsquared
print("R squared: ",r2.round(3))  # 75.1


# Residual analysis 
# -----------------

# every X variable should follow normal distribution ---> how to check 
# generally we uses histogram to check ---> if it gives bell symmetric 
# shape ---> then it is following normal distribution 

# use the huge data it is difficult so we can check through 
# residual analysis

# from the model what we fitted we can calculate 
# the errors (residuals)
model.resid   # Yactual - Ypred 

# then will these residual values we can construct histogram 

model.resid.hist()
model.resid.skew() 



# Influencer points
# -----------------

# influencing on regression line 
# It influences model fitting 

# so we need to remove these points 

# how to identify it ?? 
# through scatter plot we can plot only 2 points ---> then how ??



# Cook's Distance 
# ---------------

# calculates distance between every data point to remaining all data points
# and finds the Cook's distance 

(cooks, pvalue) = model.get_influence().cooks_distance 

df["cooks"] = pd.DataFrame(cooks)

df.head()

# plot the influencers values using stem plot 

import matplotlib.pyplot as plt 
fig = plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(df)),df["cooks"])
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()



# Leverage Value
# --------------

# we have a method to identify which are the points to be considered as 
# highly influenced points, that cutoff will take from the leverage value 


k = 2 # number of columns used for model fitting
n = len(df)

leverage_cutoff = 3*((k+1)/n)

# points which are greater than leverage value are treated as 
# influencer points 
df[df['cooks'] > leverage_cutoff] 

#      HP        MPG  VOL          SP         WT     cooks
# 70  280  19.678507   50  164.598513  15.823060  0.178098
# 76  322  36.900000   50  169.598513  16.132947  1.654415
# 78  263  34.000000   50  151.598513  15.769625  0.137228
# 79  295  19.833733  119  167.944460  39.423099  0.230841


# simply drop the rows which are recognized as highly influenced points 
df.drop([70,76,78,79],inplace=True)


#####################################################

# Again fit the model once again with out influencer points 

Y = df['MPG']
X = df[['HP','VOL']]

import statsmodels.formula.api as smf 
model = smf.ols("Y ~ X",data=df).fit()
r2 = model.rsquared
print("R square : ",r2.round(3))  # 84.4



########################################################

# Model contains R square with 75.1 including all data points 
# Model contains R square with 84.4 excluding 4 data points 

########################################################

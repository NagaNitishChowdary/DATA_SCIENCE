# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 23:49:56 2024

@author: Naga Nitish
"""

import numpy as np 
import pandas as pd 


# =======================================================

# step 1 ---> collecting the data 

df = pd.read_csv('8_Hitters_final.csv')
df

df.info()
# Data columns (total 18 columns):
#  #   Column      Non-Null Count  Dtype 
# ---  ------      --------------  ----- 
# 0   Unnamed: 0  322 non-null    object
# 1   AtBat       322 non-null    int64 
# 2   Hits        322 non-null    int64 
# 3   HmRun       322 non-null    int64 
# 4   Runs        322 non-null    int64 
# 5   RBI         322 non-null    int64 
# 6   Walks       322 non-null    int64 
# 7   Years       322 non-null    int64 
# 8   CAtBat      322 non-null    int64 
# 9   CHits       322 non-null    int64 
# 10  CHmRun      322 non-null    int64 
# 11  CRuns       322 non-null    int64 
# 12  CRBI        322 non-null    int64 
# 13  CWalks      322 non-null    int64 
# 14  PutOuts     322 non-null    int64 
# 15  Assists     322 non-null    int64 
# 16  Errors      322 non-null    int64 
# 17  Salary      322 non-null    int64 
# dtypes: int64(17), object(1)

# ====================================================


# Salary is Y variable 
# all 16 columns remaining Salary are X variables 

# and all X variables are continuous (no categorical) 

# ---> we can apply ---> multi linear regression (more than 1 X variable)
 

# =======================================================

# step 2 ---> Exploratory Data Analysis(EDA) such as 
# scatter plot,histogram, boxplot, ....

# ===> step 2 is to understand more about the data 

import matplotlib.pyplot as plt 


# scatter plots 

plt.scatter(df['Salary'],df['AtBat'])
plt.show()

df['AtBat'].hist()



plt.scatter(df['Salary'],df['Hits'])
plt.show()

df['Hits'].hist()



plt.scatter(df['Salary'],df['HmRun'])
plt.show()

df['HmRun'].hist()



plt.scatter(df['Salary'],df['Runs'])
plt.show()

df['Runs'].hist()



plt.scatter(df['Salary'],df['RBI'])
plt.show()

df['RBI'].hist()



plt.scatter(df['Salary'],df['Walks'])
plt.show()

df['Walks'].hist()



plt.scatter(df['Salary'],df['Years'])
plt.show()

df['Years'].hist()



plt.scatter(df['Salary'],df['CAtBat'])
plt.show()

df['CAtBat'].hist()



plt.scatter(df['Salary'],df['CHits'])
plt.show()

df['CHits'].hist()



plt.scatter(df['Salary'],df['CHmRun'])
plt.show()

df['CHmRun'].hist()



plt.scatter(df['Salary'],df['CRuns'])
plt.show()

df['CRuns'].hist()



plt.scatter(df['Salary'],df['CRBI'])
plt.show()

df['CRBI'].hist()



plt.scatter(df['Salary'],df['CWalks'])
plt.show()

df['CWalks'].hist()


print("scatter plot 14")
plt.scatter(df['Salary'],df['PutOuts'])
plt.show()

print("histogram 14")
print(df['PutOuts'].hist())


print("scatter plot 15")
plt.scatter(df['Salary'],df['Assists'])
plt.show()

print("histogram 15")
print(df['Assists'].hist())



print("scatter plot 16")
plt.scatter(df['Salary'],df['Errors'])
plt.show()

print("histogram 16")
print(df['Errors'].hist())

print(df.boxplot('Errors',vert=False))

# =======================================================

# step 3 ---> data cleaning

# ===============================================================

# step 4 ---> data transformation 

# apply standaradization on X variables (as they are continuous variables)

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()

X = df.iloc[:,1:17]
X_SS = SS.fit_transform(X)


Y = df['Salary']


# =========================================================

# step 5 ---> data partition

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_SS,Y,test_size = 0.30)

# ==========================================================

# step 6 ---> fit the model 

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,Y_train)

# B0 value ---> intercept 
LR.intercept_

# B1 value ---> slope
LR.coef_   

Y_Pred_train = LR.predict(X_train)
Y_Pred_test = LR.predict(X_test)

# ==============================================================

# step 8 ---> Evaluating the model using metrics 

from sklearn.metrics import mean_squared_error
training_error = mean_squared_error(Y_train,Y_Pred_train)
test_error = mean_squared_error(Y_test,Y_Pred_test)

print("Training_Error : Mean Squared Error(mse) ",training_error.round(4))
print("Training_Error : Root Mean Squared Error(RMSE) ",np.sqrt(training_error).round(4)) 

print("Test_Error : Mean Squared Error(mse) ",test_error.round(4))
print("Test_Error : Root Mean Squared Error(RMSE) ",np.sqrt(test_error).round(4))  


# ===================================================================

# STEP 9 ---> VALIDATION SET METHOD 

# Validating our model again and again for n number of times and from the 
# results, we will see which one result is highly repeated value that is 
# going to be our final value 

# Will write down our entire code in to *** for *** loop and we will 
# save the results in to a variable, then calculate the average of above value

training_err = []
test_err = [] 

for i in range(1,501):
    X_train, X_test, Y_train, Y_test = train_test_split(X_SS,Y,test_size=0.30,random_state=i)
    model = LinearRegression() 
    model.fit(X_train,Y_train)
    
    Y_Pred_train = model.predict(X_train) 
    Y_Pred_test = model.predict(X_test)
    
    training_err.append(mean_squared_error(Y_train,Y_Pred_train))
    test_err.append(mean_squared_error(Y_test,Y_Pred_test))

print("Cross_Validation : 1 -> Validation_Set_Method : Training_error " , np.mean(training_err).round(2))
print("Cross_Validation : 1 -> Validation_Set_Method : Test_error " , np.mean(test_err).round(2))

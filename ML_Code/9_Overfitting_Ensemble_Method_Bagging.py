# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 23:40:01 2024

@author: Naga Nitish
"""

# =================================================================

# step 1 ---> Collecting the data 

import numpy as np 
import pandas as pd 

df = pd.read_csv('9_Boston.csv')
df

# =================================================================

# step 2 ---> Exploratory Data Analysis(EDA) such as 
# scatter plot,histogram, boxplot, ....

# ===> step 2 is to understand more about the data 

# =================================================================

# STEP 3 ---> DATA CLEANING

# =================================================================

# STEP 4 ---> DATA TRANSFORMATION 

# apply standaradization on X variables (as they are continuous variables)

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()

X = df.iloc[:,1:14]
X_SS = SS.fit_transform(X)


Y = df.iloc[:,14]

# ==================================================================

# STEP 5 ---> DATA PARTITION 

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_SS,Y,test_size=0.30,random_state=15)

# ===================================================================

# STEP 6 ---> FIT THE MODEL 

from sklearn.linear_model import LinearRegression 
LR = LinearRegression() 
LR.fit(X_train,Y_train)

Y_Pred_train = LR.predict(X_train)
Y_Pred_test = LR.predict(X_test)

# ==========================================================================

# STEP 8 ---> EVALUATION METRICS 

from sklearn.metrics import mean_squared_error 
training_error = mean_squared_error(Y_train,Y_Pred_train)
test_error = mean_squared_error(Y_test,Y_Pred_test)

print("Training_Error : Mean Squared Error(mse) ",training_error.round(4))
print("Training_Error : Root Mean Squared Error(RMSE) ",np.sqrt(training_error).round(4)) 

print("Test_Error : Mean Squared Error(mse) ",test_error.round(4))
print("Test_Error : Root Mean Squared Error(RMSE) ",np.sqrt(test_error).round(4))  

# =========================================================================

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


# =============================================================================

# Bagging Regressor 

from sklearn.ensemble import BaggingRegressor
model_bag = BaggingRegressor(estimator=LinearRegression(), # type of model 
                             n_estimators = 100, # 100 times different models building
                             max_samples=0.6,   # 60% samples ---> randomly
                             max_features=0.7)  # 70% columns 

model_bag.fit(X_train,Y_train)

Y_Pred_train = model_bag.predict(X_train)
Y_Pred_test = model_bag.predict(X_test)

print("Bagging-Training_error: ",mean_squared_error(Y_train,Y_Pred_train))
print("Bagging-Test_error: ",mean_squared_error(Y_test,Y_Pred_test))

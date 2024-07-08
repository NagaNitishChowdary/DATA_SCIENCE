# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 22:29:47 2024

@author: Naga Nitish
"""

import numpy as np 
import pandas as pd

# =======================================================

# Step 1 ---> Collecting the data  

df = pd.read_csv('5_breast_cancer.csv')
df

# df.info()
#  #   Column           Non-Null Count  Dtype 
# ---  ------           --------------  ----- 
#  0   Unnamed: 0       683 non-null    int64 
#  1   Cl.thickness     683 non-null    int64 
#  2   Cell.size        683 non-null    int64 
#  3   Cell.shape       683 non-null    int64 
#  4   Marg.adhesion    683 non-null    int64 
#  5   Epith.c.size     683 non-null    int64 
#  6   Bare.nuclei      683 non-null    int64 
#  7   Bl.cromatin      683 non-null    int64 
#  8   Normal.nucleoli  683 non-null    int64 
#  9   Mitoses          683 non-null    int64 
#  10  Class            683 non-null    object
# dtypes: int64(10), object(1)


# ==============================================================

# step 2 ---> doing Exploratory Data Analysis (EDA) such as 
# scatter plot,histogram, boxplot, ....

# ===> step 2 is to understand more about the data 

# ===============================================================

# step 3 ---> data cleaning

# ===============================================================

# step 4 ---> data transformation 

# apply standaradization on X variables (as they are continuous variables)

from sklearn.preprocessing import StandardScaler
SS = StandardScaler() 
# df['Cl.thickness'] = SS.fit_transform(df[['Cl.thickness']])
# df['Cell.size'] = SS.fit_transform(df[['Cell.size']])
# df['Cell.shape'] = SS.fit_transform(df[['Cell.shape']])
# df['Marg.adhesion'] = SS.fit_transform(df[['Marg.adhesion']])
# df['Epith.c.size'] = SS.fit_transform(df[['Epith.c.size']])
# df['Bare.nuclei'] = SS.fit_transform(df[['Bare.nuclei']])
# df['Bl.cromatin'] = SS.fit_transform(df[['Bl.cromatin']])
# df['Normal.nucleoli'] = SS.fit_transform(df[['Normal.nucleoli']])
# df['Mitoses'] = SS.fit_transform(df[['Mitoses']])

X = df.iloc[:,1:10]
X_SS = SS.fit_transform(X)


# apply label encoding for Y variables ---> Class
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['Class_Encoded'] = LE.fit_transform(df['Class'])

Y = df['Class_Encoded'] 

df.info()
# #   Column           Non-Null Count  Dtype  
#---  ------           --------------  -----  
# 0   Unnamed: 0       683 non-null    int64  
# 1   Cl.thickness     683 non-null    float64
# 2   Cell.size        683 non-null    float64
# 3   Cell.shape       683 non-null    float64
# 4   Marg.adhesion    683 non-null    float64
# 5   Epith.c.size     683 non-null    float64
# 6   Bare.nuclei      683 non-null    float64
# 7   Bl.cromatin      683 non-null    float64
# 8   Normal.nucleoli  683 non-null    float64
# 9   Mitoses          683 non-null    float64
# 10  Class            683 non-null    int32  
# dtypes: float64(9), int32(1), int64(1)

# =========================================================

# step 5 ---> data partition

from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X_SS,Y,test_size = 0.30)

# ==========================================================

# step 6 ---> fit the model 

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train,Y_train)


Y_Pred_train = model.predict(X_train)
Y_Pred_test = model.predict(X_test)

# ==========================================================

# step 8 ---> Evaluating the model using metrics 

from sklearn.metrics import accuracy_score

print("Training Accuracy Score: ", accuracy_score(Y_train,Y_Pred_train).round(2))
print("Testing Accuracy Score: ",accuracy_score(Y_test,Y_Pred_test).round(2))


# ============================================================

# CROSS VALIDATION 

# X_train, X_test, Y_train, Y_test = train_test_split(X_SS,Y,test_size = 0.30)
# train_test_split(....) function randomly distributes the data, so 
# the accuracy slightly changes when runs multiple times 

# To fix it , we use random_state ---> it can take any value, 
# the specific number doesn't matter


# Is it correct to fix only some particular values for training 

# what if the values are baised ---> i.e. all selected samples are different from others 

# so, we apply cross-validation technique on it  

# using random_state
#                    training_accuracy  test_accuracy 
# random state = 16       90                99 
#              = 14       98                85
#              = 10       90                70


# ================================================================

# 1 ---> Validation Set Method 

# Validating our model again and again for n number of times and from the 
# results, we will see which one result is highly repeated value that is 
# going to be our final value 

# Will write down our entire code in to *** for *** loop and we will 
# save the results in to a variable, then calculate the average of above value 

training_accuracy = [] 
test_accuracy = [] 

for i in range(1,501):
    X_train, X_test, Y_train, Y_test = train_test_split(X_SS,Y,test_size=0.30,random_state=i)
    model = LogisticRegression()
    model.fit(X_train,Y_train)
    Y_Pred_train = model.predict(X_train)
    Y_Pred_test = model.predict(X_test)
    
    training_accuracy.append(accuracy_score(Y_train,Y_Pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_Pred_test))
    
print("Cross_Validation : 1 -> Validation_Set_Method : Training_accuracy " , np.mean(training_accuracy).round(2))
print("Cross_Validation : 1 -> Validation_Set_Method : Testing_accuracy ",np.mean(test_accuracy).round(2))
print()

# ==================================================================

# 2 ---> K-Fold Cross Validation 

# Entire data is divided into K Folders 

# e.g. 1008 samples where k = 5 

#  samples   1     2      3      4      5       
# 1. 202   test  train  train  train  train
# 2. 202   train test   train  train  train
# 3. 202   train train  test   train  train
# 4. 201   train train  train  test   train
# 5. 201   train train  train  train  test

# when we have huge amount of data, it will be very useful 

# every folders is in training as well as in testing --> covering whole data

from sklearn.model_selection import KFold
kf = KFold(n_splits=5)

training_accuracy = [] 
test_accuracy = [] 

for train_index,test_index in kf.split(X_SS):
    X_train, X_test, Y_train, Y_test = X_SS[train_index] , X_SS[test_index] , Y.iloc[train_index] , Y.iloc[test_index]
    model.fit(X_train,Y_train)
    Y_Pred_train = model.predict(X_train)
    Y_Pred_test = model.predict(X_test)
    
    training_accuracy.append(accuracy_score(Y_train,Y_Pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_Pred_test))
    
print("Cross_Validation : 2 -> K-FOLD_CROSS_VALIDATION : Training_accuracy " , np.mean(training_accuracy).round(2))
print("Cross_Validation : 2 -> K-FOLD_CROSS_VALIDATION : Testing_accuracy ",np.mean(test_accuracy).round(2))

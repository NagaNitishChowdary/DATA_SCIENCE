# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 08:40:00 2024

@author: Naga Nitish
"""

import numpy as np 
import pandas as pd 

df = pd.read_csv('8_Hitters_final.csv')
df

Y = df['Salary']
X = df.iloc[:,1:17] # ---> 0 th col is Unnamed 

# ===========================================================

# DATA TRANSFORMATION 

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()

# list(X)  --> gives column names
cols = list(X)

X = SS.fit_transform(X)
X = pd.DataFrame(X)
X.columns = cols

# ============================================================

# Data Partition 

from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30,random_state=1)


# ================================================================

# Model Fitting

from sklearn.linear_model import LinearRegression 
LR = LinearRegression()

LR.fit(X_train,Y_train)

LR_coef = LR.coef_
# print(LR_coef)
# [-235.14852401  342.09066131  115.37039899  -80.15738256 -119.25798751
#  112.52780959  -44.77407125 -282.30461651  169.69097496  -71.77805652
#  226.48911615  283.93587877 -136.15442918   53.2115293    15.41874618
#  -12.84231165]

Y_Pred_train = LR.predict(X_train)
Y_Pred_test = LR.predict(X_test)

# =================================================================

# METRICS 

from sklearn.metrics import mean_squared_error 
print("Regularization ---> Training root_mean_squared_error : ", mean_squared_error(Y_Pred_train,Y_train,squared=False).round(2))
print("Regularization ---> Test root_mean_squared_error : ", mean_squared_error(Y_Pred_test,Y_test,squared=False).round(2)) 


# Training_accuracy ---> 294.25
# Test_accuracy ---> 337.03 

# ==================================================================


# Data is randomly divided so we cannot trust so apply cross validation technique 

# VALIDATION SET METHOD 

training_acc = [] 
test_acc = [] 

for i in range(1,501):
    X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=i)
    LR = LinearRegression() 
    LR.fit(X_train,Y_train)
    
    Y_Pred_train = LR.predict(X_train)
    Y_Pred_test = LR.predict(X_test)
    
    training_acc.append(mean_squared_error(Y_Pred_train,Y_train,squared=False).round(2))
    test_acc.append(mean_squared_error(Y_Pred_test,Y_test,squared=False).round(2))
    
print("Regularization ---> Validation_Set_Method ---> Training_rmse : ",np.mean(training_acc).round(2))
print("Regularization ---> Validation_Set_Method ---> Test_rmse : ",np.mean(test_acc).round(2))


# Training_accuracy ---> 298.39 
# Test_accuracy ---> 337.03 

# ====================================================================


# RIGID REGRESSION 

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=1)

from sklearn.linear_model import Ridge 
R = Ridge(alpha=1)

R.fit(X_train,Y_train)

R_coef = R.coef_
# print(R_coef)
# [-213.45028588  293.28865873   86.68075966  -50.17591503  -88.25451219
#  102.76876009  -60.10578513 -122.10224264  158.47102909  -15.52339591
#  135.20582539  178.23318814 -124.12621683   54.78808039    8.21424567
#  -12.63565888]

d1 = pd.DataFrame(cols)
d2 = pd.DataFrame(LR_coef)
d3 = pd.DataFrame(R_coef) 

pd.concat([d1,d2,d3],axis=1)   # for alpha = 1.0 
#          0           0           0
# 0     AtBat -235.148524 -213.450286
# 1      Hits  342.090661  293.288659
# 2     HmRun  115.370399   86.680760
# 3      Runs  -80.157383  -50.175915
# 4       RBI -119.257988  -88.254512
# 5     Walks  112.527810  102.768760
# 6     Years  -44.774071  -60.105785
# 7    CAtBat -282.304617 -122.102243
# 8     CHits  169.690975  158.471029
# 9    CHmRun  -71.778057  -15.523396
# 10    CRuns  226.489116  135.205825
# 11     CRBI  283.935879  178.233188
# 12   CWalks -136.154429 -124.126217
# 13  PutOuts   53.211529   54.788080
# 14  Assists   15.418746    8.214246
# 15   Errors  -12.842312  -12.635659

#    count  length_of_digits 
# LR  10   --->   3 
#      6   --->   2

# Ridge  8 ---> 3
#        7 ---> 2 
#        1 ---> 1  


# for alpha = 0.0  ---> no change in LR_coef and Ridge_coefficients 
#          0           0           0
# 0     AtBat -235.148524 -235.148524
# 1      Hits  342.090661  342.090661
# 2     HmRun  115.370399  115.370399
# 3      Runs  -80.157383  -80.157383
# 4       RBI -119.257988 -119.257988
# 5     Walks  112.527810  112.527810
# 6     Years  -44.774071  -44.774071
# 7    CAtBat -282.304617 -282.304617
# 8     CHits  169.690975  169.690975
# 9    CHmRun  -71.778057  -71.778057
# 10    CRuns  226.489116  226.489116
# 11     CRBI  283.935879  283.935879
# 12   CWalks -136.154429 -136.154429


# for alpha = 10.0 
#          0           0           0
# 0     AtBat -235.148524  -85.326628
# 1      Hits  342.090661  133.691122
# 2     HmRun  115.370399   30.297368
# 3      Runs  -80.157383    5.945006
# 4       RBI -119.257988  -23.538512
# 5     Walks  112.527810   64.729601
# 6     Years  -44.774071  -49.498318
# 7    CAtBat -282.304617    6.957766
# 8     CHits  169.690975   93.803713
# 9    CHmRun  -71.778057   21.665245
# 10    CRuns  226.489116   59.122493
# 11     CRBI  283.935879   85.399522
# 12   CWalks -136.154429  -61.245409
# 13  PutOuts   53.211529   57.069357
# 14  Assists   15.418746   -8.871080
# 15   Errors  -12.842312  -12.216227

#    count  length_of_digits 
# LR  10   --->   3 
#      6   --->   2

# Ridge  1  --->  3 
#        12 ---> 2


# for alpha = 20.0   
#          0           0          0
# 0     AtBat -235.148524 -43.436611
# 1      Hits  342.090661  91.058719
# 2     HmRun  115.370399  18.954170
# 3      Runs  -80.157383  15.200535
# 4       RBI -119.257988  -6.684236
# 5     Walks  112.527810  51.779265
# 6     Years  -44.774071 -35.287482
# 7    CAtBat -282.304617  17.716601
# 8     CHits  169.690975  71.056957
# 9    CHmRun  -71.778057  25.222960
# 10    CRuns  226.489116  47.764547
# 11     CRBI  283.935879  65.401138
# 12   CWalks -136.154429 -34.846822
# 13  PutOuts   53.211529  56.005286
# 14  Assists   15.418746 -12.887074
# 15   Errors  -12.842312 -11.502350

#    count  length_of_digits 
# LR  10   --->   3 
#      6   --->   2

# Ridge  0  --->  3 
#        15 ---> 2



# for alpha = 500  
#          0           0          0
# 0     AtBat -235.148524  14.936506
# 1      Hits  342.090661  22.050812
# 2     HmRun  115.370399  14.932841
# 3      Runs  -80.157383  17.103947
# 4       RBI -119.257988  17.629886
# 5     Walks  112.527810  20.391112
# 6     Years  -44.774071   9.849908
# 7    CAtBat -282.304617  18.582135
# 8     CHits  169.690975  22.052937
# 9    CHmRun  -71.778057  19.144238
# 10    CRuns  226.489116  20.897179
# 11     CRBI  283.935879  22.235438
# 12   CWalks -136.154429  13.621929
# 13  PutOuts   53.211529  22.689062
# 14  Assists   15.418746  -5.410574
# 15   Errors  -12.842312  -2.319077

# HERE FOR ALPHA 500 ALSO COEFFICIENTS ARE DECREASING BUT NOT TENDING TO ZERO.
# SO FOR THIS PROBLEM, RIDGE REGRESSION IS NOT THE GREAT FIT. 

# ======================================================================

# ====> WHEN ALPHA IS INCREASING THEN THE COEFFICIENTS ARE TENDING TO ZERO

# ======================================================================


# LASSO REGRESSION 

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=1)

from sklearn.linear_model import Lasso 
L = Lasso(alpha=1.0)

L.fit(X_train,Y_train)

L_coef = L.coef_
# print(L_coef)
# [-228.81565708  294.6462018    68.46341124  -33.20361876  -69.625731
#    97.03121766  -67.35372533   -0.           91.35633137    0.
#    96.57780586  149.29900599 -117.62857065   54.87133559    0.35461328
#    -7.56808436]

d1 = pd.DataFrame(cols)
d2 = pd.DataFrame(LR_coef)
d3 = pd.DataFrame(R_coef) 
d4 = pd.DataFrame(L_coef)

pd.concat([d1,d2,d3,d4],axis=1)

# for alpha = 1 
#           0           0           0           0
# 0     AtBat -235.148524 -213.450286 -228.815657
# 1      Hits  342.090661  293.288659  294.646202
# 2     HmRun  115.370399   86.680760   68.463411
# 3      Runs  -80.157383  -50.175915  -33.203619
# 4       RBI -119.257988  -88.254512  -69.625731
# 5     Walks  112.527810  102.768760   97.031218
# 6     Years  -44.774071  -60.105785  -67.353725
# ---> 7    CAtBat -282.304617 -122.102243   -0.000000
# 8     CHits  169.690975  158.471029   91.356331
# ---> 9    CHmRun  -71.778057  -15.523396    0.000000
# 10    CRuns  226.489116  135.205825   96.577806
# 11     CRBI  283.935879  178.233188  149.299006
# 12   CWalks -136.154429 -124.126217 -117.628571
# 13  PutOuts   53.211529   54.788080   54.871336
# 14  Assists   15.418746    8.214246    0.354613
# 15   Errors  -12.842312  -12.635659   -7.568084

# For alpha = 1 ---> 2 variables are shrinking to zero ---> so we can tell that 
# LASSO Is working good 

# SO DROP COLUMNS 7,9 AND FIT AGAIN 

X_new = X.drop(['CAtBat','CHmRun'],axis=1)

# VALIDATION SET METHOD

training_acc1 = []
test_acc1 = [] 

for i in range(1,501):
    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X_new,Y,test_size=0.3,random_state=i)
    model = LinearRegression()
    model.fit(X_train1,Y_train1)
    Y_Pred_train1 = model.predict(X_train1)
    Y_Pred_test1 = model.predict(X_test1)
    
    training_acc1.append(mean_squared_error(Y_Pred_train1,Y_train1,squared=False).round(2))
    test_acc1.append(mean_squared_error(Y_Pred_test1,Y_test1,squared=False).round(2))
    
print("After Applying Lasso ---> Validation_Set_Method ---> Training_rmse : ",np.mean(training_acc1).round(2))
print("After Applying Lasso ---> Validation_Set_Method ---> Test_rmse : ",np.mean(test_acc1).round(2))

# Training_acc = 300.35
# Test_acc = 332.64

# ===================================================================

#   alpha           drop  train    test    diff
#    0      linear   0    298.39   337.03  38.64
#    1      lasso    2    300.35   332.64  32.29 ---> diff is decresed
#    3      lasso    3    300.7    330.89  30.19 ---> diff is decresed
#    4      lasso    5    304.83   331.14  26.31 ---> diff is decreased but test error is increasing , we need to control it 
# but 2 variables dropping is greater than slight increase of test error(0.25)
#    5      lasso    6    306.61   331.38  24.77 ---> diff is decreased but test error is increasing


# ====================================================================

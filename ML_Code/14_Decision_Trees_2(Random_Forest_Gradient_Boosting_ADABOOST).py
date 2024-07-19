# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 18:06:05 2024

@author: Naga Nitish
"""

import numpy as np 
import pandas as pd 

# STEP 1 ---> IMPORT THE DATA 

df = pd.read_csv('14_sales.csv')
df


# STEP 2 ---> EDA 

# STEP 3 ---> DATA CLEANING 

# STEP 4 ---> DATA TRANSFORMATION 

df.info()
"""
RangeIndex: 400 entries, 0 to 399
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype 
---  ------       --------------  ----- 
 0   Unnamed: 0   400 non-null    int64 
 1   CompPrice    400 non-null    int64 
 2   Income       400 non-null    int64 
 3   Advertising  400 non-null    int64 
 4   Population   400 non-null    int64 
 5   Price        400 non-null    int64 
 6   ShelveLoc    400 non-null    object
 7   Age          400 non-null    int64 
 8   Education    400 non-null    int64 
 9   Urban        400 non-null    object
 10  US           400 non-null    object
 11  high         400 non-null    object
"""

df_categorical_var = ['ShelveLoc','Urban','US']
df_num = ['CompPrice','Income','Advertising','Population','Price','Age']


# FOR CATEGORICAL VARIABLES PERFORM LABEL ENCODING 

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder() 

for i in df_categorical_var:
    df[i] = LE.fit_transform(df[i])
    

# FOR NUMBERICAL VARIABLES PERFORM STANDARDIZATION

from sklearn.preprocessing import StandardScaler
SS = StandardScaler() 

for i in df_num:
    df[i] = SS.fit_transform(df[[i]])
    

Y = df['high']
X = df[['CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban','US']]
X

# STEP 5 ---> DATA PARTITION 

from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)


# STEP 6 ---> FITTING THE MODEL 

from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier() 

DTC.fit(X_train, Y_train)

Y_Pred_train = DTC.predict(X_train)
Y_Pred_test = DTC.predict(X_test)


from sklearn.metrics import accuracy_score 
print("Decision_Tree ---> Training_accuracy : ",accuracy_score(Y_train,Y_Pred_train).round(2))
print("Decision_Tree ---> Test_accuracy : ",accuracy_score(Y_test,Y_Pred_test).round(2))


# Visualizing the decision tree 
from sklearn import tree 
import matplotlib.pyplot as plt 
plt.figure(figsize=(15,10))
tree.plot_tree(DTC,filled=True)

node_count = DTC.tree_.node_count 
max_depth = DTC.tree_.max_depth 
print("Number of nodes: ",node_count)  # 101
print("Depth of Tree: ",max_depth)  # 10 

# HERE TRAINING ACCURACY IS 100%
# TESTING ACCURACY IS 70%

# SO MUCH DIFFERENCE BETWEEN TRAINING AND TESTING ACCURACY , 
# SO VARIABLES MAY BE ADDED UNNECCESRILY ---> MODEL OVERFITTED 

from sklearn.ensemble import BaggingClassifier 
bag_clf = BaggingClassifier(estimator = DTC,
                            n_estimators=100,
                            max_samples=0.7,
                            max_features=0.8,
                            random_state=42)

bag_clf.fit(X_train, Y_train)

Y_Pred_train = bag_clf.predict(X_train)
Y_Pred_test = bag_clf.predict(X_test)

print("Decision_Trees --> Bagging_Classifier ---> Training_accuracy : ",accuracy_score(Y_Pred_train,Y_train).round(2))
print("Decision_Trees --> Bagging_Classifier ---> Test_accuracy : ",accuracy_score(Y_Pred_test,Y_test).round(2)) 

# Training accuracy ===> 100%
# Testing accuracy ===> 82%  ---> some what improved in controlling overfitted model 

# ===============================================================================


# Random Forest (same as Bagging (BootStrap Aggregation)) but only applicable 
# to Decision Trees 

from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100 , max_samples=0.7, max_features = 0.8, random_state = 42)
RFC.fit(X_train,Y_train)

Y_Pred_train = RFC.predict(X_train)
Y_Pred_test = RFC.predict(X_test)

print("Decision_Trees --> Random_Forest ---> Training_accuracy : ",accuracy_score(Y_Pred_train,Y_train))
print("Decision_Trees --> Random_Forest ---> Test_accuracy : ",accuracy_score(Y_Pred_test,Y_test))

# Training_accuracy = 99%
# Test_accuracy = 81.25%  ---> bagging is doing better than random forest 



# =======================================================================

# SEQUENCIAL BAGGING METHODS 

# =======================================================================

# GRADIENT BOOSTING CLASSIFIER 

from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier(n_estimators=100,max_features=0.8,random_state=42,learning_rate=0.1)
GBC.fit(X_train,Y_train)

Y_Pred_train = GBC.predict(X_train)
Y_Pred_test = GBC.predict(X_test) 

print("Decision_Trees --> Gradient_Boosting ---> Training_accuracy : ",accuracy_score(Y_Pred_train,Y_train).round(2))
print("Decision_Trees --> Gradient_Boosting ---> Test_accuracy : ",accuracy_score(Y_Pred_test,Y_test).round(2))

# Training_accuracy ===> 100% 
# Test_accuracy ===> 90% ---> far better than bagging and random forest


# =====================================================================


# ADA_BOOST --> Adaptive Boosting Technique 

# focused on samples which are incorrectly predicted on the previous set of examples. 

from sklearn.ensemble import AdaBoostClassifier
ABC = AdaBoostClassifier(n_estimators=100,random_state=42)

ABC.fit(X_train,Y_train)

Y_Pred_train = ABC.predict(X_train)
Y_Pred_test = ABC.predict(X_test)

print("Decision_Trees ---> AdaBoost ---> Training_accuracy : ",accuracy_score(Y_Pred_train,Y_train).round(2))
print("Decision_Trees ---> AdaBoost ---> Test_accuracy : ",accuracy_score(Y_Pred_test,Y_test).round(2))

# Training_accuracy ===> 100%
# Test_accuracy ===> 88%    ----> Same as Gradient Boosting

# =========================================================================

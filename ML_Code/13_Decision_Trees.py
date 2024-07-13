# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 01:21:25 2024

@author: Naga Nitish
"""

import numpy as np 
import pandas as pd 

# COLLECTING THE DATA 

df = pd.read_csv('12_13_mushroom.csv')
df


# EDA 

# DATA CLEANING 

# DATA TRANSFORMATION 

df.info()
#RangeIndex: 8416 entries, 0 to 8415
#Data columns (total 23 columns):
# #   Column                 Non-Null Count  Dtype 
#---  ------                 --------------  ----- 
# 0   Typeofmushroom         8416 non-null   object
# 1   capshape               8416 non-null   object
# 2   capsurface             8416 non-null   object
# 3   capcolor               8416 non-null   object
# 4   bruises                8416 non-null   object
# 5   odor                   8416 non-null   object
# 6   gillattachment         8416 non-null   object
# 7   gillspacing            8416 non-null   object
# 8   gillsize               8416 non-null   object
# 9   gillcolor              8416 non-null   object
# 10  stalkshape             8416 non-null   object
# 11  stalkroot              8416 non-null   object
# 12  stalksurfaceabovering  8416 non-null   object
# 13  stalksurfacebelowring  8416 non-null   object
# 14  stalkcolorabovering    8416 non-null   object
# 15  stalkcolorbelowring    8416 non-null   object
# 16  veiltype               8416 non-null   object
# 17  veilcolor              8416 non-null   object
# 18  ringnumber             8416 non-null   object
# 19  ringtype               8416 non-null   object
# 20  sporeprintcolor        8416 non-null   object
# 21  population             8416 non-null   object
# 22  habitat                8416 non-null   object

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

for i in range(0,23):
    df.iloc[:,i] = LE.fit_transform(df.iloc[:,i])
    
    
# AFTER TRANSFORMATION ALSO THERE ARE IN OBJECT DATA TYPE 

for col in df.columns:
    df[col] = df[col].astype(int)
    

# DATA PARTITION 

Y = df['Typeofmushroom']
X = df.iloc[:,1:]

from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y) 


# Fitting the model with Decision Tree Classifier 

from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()

DTC.fit(X_train,Y_train)

Y_Pred_train = DTC.predict(X_train)
Y_Pred_test = DTC.predict(X_test)

print("Decision_Trees ---> Training_accuracy : ",accuracy_score(Y_Pred_train,Y_train).round(2))
print("Decision_Trees --> Test_accuracy",accuracy_score(Y_Pred_test,Y_test).round(2))


# Training_acc = 1.0 ==> 100%
# Testing_acc = 1.0 ===> 100%

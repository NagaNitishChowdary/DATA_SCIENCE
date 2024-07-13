# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 16:50:45 2024

@author: Naga Nitish
"""

import numpy as np 
import pandas as pd 

# COLLECTING THE DATA 

df = pd.read_csv('12_mushroom.csv')
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

# FITTING THE MODEL 

from sklearn.naive_bayes import MultinomialNB 
MNB = MultinomialNB()

MNB.fit(X_train,Y_train)

Y_Pred_train = MNB.predict(X_train)
Y_Pred_test = MNB.predict(X_test)

# EVALUATION METRICS 

from sklearn.metrics import accuracy_score 
print("Naive_Bayes --> Training_accuracy : ",accuracy_score(Y_train,Y_Pred_train).round(2))
print('Naive Bayes --> Test_accuracy : ',accuracy_score(Y_test,Y_Pred_test).round(2))


# CROSS - VALIDATION 

# VALIDATION SET METHOD 

training_acc = []
test_acc = []

for i in range(1,101):
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    model = MultinomialNB()
    model.fit(X_train,Y_train)
    Y_Pred_train = model.predict(X_train)
    Y_Pred_test = model.predict(X_test)
    
    training_acc.append(accuracy_score(Y_Pred_train,Y_train))
    test_acc.append(accuracy_score(Y_Pred_test,Y_test))
    
print("Naive_Bayes --> Validation_Set_Method ---> Training_accuracy : ",np.mean(training_acc).round(2))
print("Naive_Bayes --> Validation_Set Method ---> Test_accuracy : ",np.mean(test_acc).round(2))

# Training_acc = 83 
# Test_accuracy = 82 


# Lets try with bagging classifier (because target variable is related to classification)

from sklearn.ensemble import BaggingClassifier 
model_bag = BaggingClassifier(estimator = MultinomialNB(),
                              n_estimators=200,
                              max_samples=0.8,
                              max_features=0.9)

model_bag.fit(X_train,Y_train)

Y_Pred_train = model_bag.predict(X_train)
Y_Pred_test = model_bag.predict(X_test)

print("Naive_Bayes --> Bagging_Classifier ---> Training_accuracy : ",accuracy_score(Y_Pred_train,Y_train).round(2))
print("Naive_Bayes --> Bagging_Classifier ---> Test_accuracy : ",accuracy_score(Y_Pred_test,Y_test).round(2))

# Training_Acc = 0.82 
# Test_acc = 0.84


# Accuracy is some what low so try it with Decision Tree Classifier 

from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()

DTC.fit(X_train,Y_train)

Y_Pred_train = DTC.predict(X_train)
Y_Pred_test = DTC.predict(X_test)

print("Decision_Trees ---> Training_accuracy : ",accuracy_score(Y_Pred_train,Y_train).round(2))
print("Decision_Trees --> Test_accuracy",accuracy_score(Y_Pred_test,Y_test).round(2))


# Training_acc = 1.0 ==> 100%
# Testing_acc = 1.0 ===> 100%

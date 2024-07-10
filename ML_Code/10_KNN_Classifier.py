# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:27:21 2024

@author: Naga Nitish
"""

import numpy as np 
import pandas as pd 

df = pd.read_csv('10_breast-cancer-wisconsin-data.csv')
df


Y = df['diagnosis']
X = df.iloc[:,2:]

# ==========================================================

from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y) 

# ==========================================================

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, Y_train)

Y_Pred_train = knn.predict(X_train)
Y_Pred_test = knn.predict(X_test)

# ============================================================

from sklearn.metrics import accuracy_score
print("KNN: Training_accuracy: ",accuracy_score(Y_train,Y_Pred_train).round(2))
print("KNN: Test_accuracy: ",accuracy_score(Y_test,Y_Pred_test).round(2))

# The testing and training accuracy changes when running multiple times 
# because train_test_split ---> takes randomly values every time running

# ===============================================================

# So applying VALIDATION SET METHOD 

training_accuracy_list = []
test_accuracy_list = []


# We always takes K values as odd 
for k in range(5,18,2):
    training_acc = [] 
    test_acc = [] 
    
    # for i in range(1,100):
    for i in range(1,3):
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train.values, Y_train)
        Y_Pred_train = knn.predict(X_train.values)
        Y_Pred_test = knn.predict(X_test.values)
        
        training_acc.append(accuracy_score(Y_train,Y_Pred_train))
        test_acc.append(accuracy_score(Y_test,Y_Pred_test)) 

    training_accuracy_list.append(np.mean(training_acc).round(2))
    test_accuracy_list.append(np.mean(test_acc).round(2))
    
print("Training accuracy list : ",training_accuracy_list)
print("Testing accuracy list: ",test_accuracy_list)


# ===============================================================

import matplotlib.pyplot as plt 
plt.scatter(range(5,18,2),training_accuracy_list,color='blue')
plt.plot(range(5,18,2),training_accuracy_list,color='black')
plt.scatter(range(5,18,2),test_accuracy_list,color='red')
plt.plot(range(5,18,2),test_accuracy_list,color='black')
plt.show()

# ===> here it is showing at from K = 7 , there is least variance between 
# training_accuracy and test_accuracy  

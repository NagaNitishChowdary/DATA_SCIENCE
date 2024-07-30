# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 01:14:31 2024

@author: Naga Nitish
"""

import numpy as np 
import pandas as pd 

df = pd.read_csv('10_breast-cancer-wisconsin-data.csv')
df


Y = df['diagnosis']
X = df.iloc[:,2:]

# =============================================================

# DATA TRANSFORMATION  

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)


# ==============================================================

# PRINCIPAL COMPONENT ANALYSIS

from sklearn.decomposition import PCA 
pca = PCA()

PC = pca.fit_transform(SS_X)
PC = pd.DataFrame(PC)
PC

# NOW FIT MODEL ONCE AGAIN USING FIRST FEW COMPONENTS 

X_new = PC.iloc[:, 0:8]   # Taking 8 Principal components(cols) as input


# ================================================================

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

training_accuracy_list = []
test_accuracy_list = [] 

for k in range(5,18,2):
    training_acc = []
    test_acc = [] 
    
    for i in range(1,100):
        X_train, X_test, Y_train, Y_test = train_test_split(X_new,Y,test_size=0.30,random_state = i)
        knn = KNeighborsClassifier(n_neighbors=k)    
        knn.fit(X_train,Y_train)
        
        Y_Pred_train = knn.predict(X_train)
        Y_Pred_test = knn.predict(X_test)
        
        training_acc.append(accuracy_score(Y_train,Y_Pred_train))
        test_acc.append(accuracy_score(Y_test,Y_Pred_test))
        
    training_accuracy_list.append(np.mean(training_acc).round(2)) 
    test_accuracy_list.append(np.mean(test_acc).round(2))
    
print("Training accuries: ",training_accuracy_list)
print("Testing accuracies: ",test_accuracy_list)


import matplotlib.pyplot as plt 
plt.scatter(range(5,18,2),training_accuracy_list,color='blue')
plt.plot(range(5,18,2),training_accuracy_list,color='black')
plt.scatter(range(5,18,2),test_accuracy_list,color='red')
plt.plot(range(5,18,2),test_accuracy_list,color='black')
plt.show()

# ============================================================

# Using all 30 X variables 
# training_acc = 94% 
# test_acc = 93%

# For 8 Principle components, 
# training_acc = 97%
# test_acc = 96% 


# For 7 Priciple Components, 
# training_acc = 96%
# test_acc = 96%


# For 6 PC's, 
# training_acc = 97%
# test_acc = 96%


# For 5 PC's
# training_acc = 97% 
# test_acc =  96%

# For 4 PC's 
# training_acc = 96% 
# test_acc = 96%


# For 4 variables also , we are getting the best results 
# Lower Complexity with better results 

# This is how we utilize the priciple components 

# **** This can be worked out only on continuous variables 



# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 22:42:37 2024

@author: Naga Nitish
"""

# ==============================================================

import numpy as np 
import pandas as pd 

df = pd.read_csv('11_createdata.csv')
df


# =============================================================

Y = df['Y']


# HERE TO VISUALIZE THE GRAPH WE HAVE ONLY TAKEN 2 VARIABLES 
# BUT IN REALITY, WE NEED TO TAKE ALL THE VARIABLES 

X = df.iloc[:,1:3]
X

# =============================================================

# data partition 

from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

# =============================================================

# fitting the model 

from sklearn.svm import SVC    # SVC --> support vector classifier
model = SVC(kernel = 'linear')  

model.fit(X_train, Y_train)

Y_Pred_train = model.predict(X_train) 
Y_Pred_test = model.predict(X_test)


from sklearn.metrics import accuracy_score 
print('SVM -> linear : Training_accuracy: ',accuracy_score(Y_train,Y_Pred_train).round(2))
print('SVM -> linear : Test_accuracy: ',accuracy_score(Y_test,Y_Pred_test).round(2))

# training_accuracy = 92 
# test_accuracy = 91 

# ploting and showing the line that seperated two classes 

# from mlxtend.plotting import plot_decision_regions 
# plot_decision_regions(X = X, Y=Y, clf=model, legend=4)

# ===================================================================

# fitting the model (polynomial)

from sklearn.svm import SVC    # SVC --> support vector classifier
model = SVC(kernel = 'poly', degree=5)


# degree - 2 ---> training_accuracy = 50% 
# degree - 3 ---> training_accuracy = 92%
# degree - 4 ---> training_accuracy = 51%
# degree - 5 ---> training_accuracy = 87%
 

# it seems like even numbers are not working 

model.fit(X_train, Y_train)

Y_Pred_train = model.predict(X_train) 
Y_Pred_test = model.predict(X_test)


from sklearn.metrics import accuracy_score 
print('SVM -> polynomial : Training_accuracy: ',accuracy_score(Y_train,Y_Pred_train).round(2))
print('SVM -> polynomial : Test_accuracy: ',accuracy_score(Y_test,Y_Pred_test).round(2))


# ploting and showing the line that seperated two classes 

# from mlxtend.plotting import plot_decision_regions 
# plot_decision_regions(X = X, Y=Y, clf=model, legend=4)


# ===================================================================

# fitting the model (radial bais function ---> rbf)

from sklearn.svm import SVC    # SVC --> support vector classifier
model = SVC(kernel = 'rbf') 

model.fit(X_train, Y_train)

Y_Pred_train = model.predict(X_train) 
Y_Pred_test = model.predict(X_test)


from sklearn.metrics import accuracy_score 
print('SVM -> radial_bais_function : Training_accuracy: ',accuracy_score(Y_train,Y_Pred_train).round(2))
print('SVM -> radial_bais_function : Test_accuracy: ',accuracy_score(Y_test,Y_Pred_test).round(2))


# accuracy ---> training -> 92 , testing ---> 91 

# ploting and showing the line that seperated two classes 

# from mlxtend.plotting import plot_decision_regions 
# plot_decision_regions(X = X, Y=Y, clf=model, legend=4)

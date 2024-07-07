# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 16:27:17 2024

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

# ==========================================================

# step 6 ---> fit the model 

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_SS,Y)
Y_Pred = model.predict(X_SS)


# ==========================================================

# step 8 ---> Evaluating the model using metrics 

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score,precision_score,f1_score

cm = confusion_matrix(Y,Y_Pred)
print("Confusion Matrix: " , cm)

accuracy = accuracy_score(Y,Y_Pred)
print("Accuracy : ", accuracy.round(2))  # 0.97

sensitivity = recall_score(Y,Y_Pred)
print("sensitivity : ", sensitivity.round(2))  # 0.96

ps = precision_score(Y,Y_Pred)
print("Precision Score : ", ps.round(2))  # 0.96

f1 = f1_score(Y,Y_Pred)
print("F1 Score : ", f1.round(2))  # 0.96

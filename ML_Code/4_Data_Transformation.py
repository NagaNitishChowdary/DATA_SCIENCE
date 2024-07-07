# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 19:05:24 2024

@author: Naga Nitish
"""


# STEP 5 IN PROJECT LIFE CYCLE ---> DATA TRANSFORMATION 


import numpy as np 
import pandas as pd 

df = pd.read_csv('4_healthcare-dataset-stroke-data.csv') 
df 


df.info()

#  #   Column             Non-Null Count  Dtype  
# ---  ------             --------------  -----  
#  0   id                 4799 non-null   int64  
#  1   gender             4799 non-null   object 
#  2   age                4799 non-null   int64  
#  3   hypertension       4799 non-null   int64  
#  4   heart_disease      4799 non-null   int64  
#  5   ever_married       4799 non-null   object 
#  6   work_type          4799 non-null   object 
#  7   Residence_type     4799 non-null   object 
#  8   avg_glucose_level  4799 non-null   float64
#  9   bmi                4799 non-null   float64
#  10  smoking_status     4799 non-null   object 
#  11  stroke             4799 non-null   int64  
# dtypes: float64(2), int64(5), object(5)


# Some of the X variables are continuous and some are of discrete 
# then you need to do data transformation 


# Data transformation
 
# ---> If X variables are continuous variables , then do Standardization 
# ---> If X variables are categorical variables, then do Label Encoding 


# Label Encoding  
# gender , ever_married , work_type , Residence_type , smoking_status 

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder() 
df['gender'] = LE.fit_transform(df['gender'])
df['ever_married'] = LE.fit_transform(df['ever_married'])
df['work_type'] = LE.fit_transform(df['work_type'])
df['Residence_type'] = LE.fit_transform(df['Residence_type'])
df['smoking_status'] = LE.fit_transform(df['smoking_status'])


# Standardization 
# age , avg_glucose_level , bmi 

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()

df['age'] = SS.fit_transform(df[['age']])
df['bmi'] = SS.fit_transform(df[['bmi']])
df['avg_glucose_level'] = SS.fit_transform(df[['avg_glucose_level']])

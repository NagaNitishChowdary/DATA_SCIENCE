# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 22:19:57 2024

@author: Naga Nitish
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv('23_daily-minimum-temperatures.csv')
df

#             Date  Temp
# 0     1981-01-01  20.7
# 1     1981-01-02  17.9
# 2     1981-01-03  18.8
# 3     1981-01-04  14.6
# 4     1981-01-05  15.8
#          ...   ...
# 3645  1990-12-27  14.0
# 3646  1990-12-28  13.6
# 3647  1990-12-29  13.5
# 3648  1990-12-30  15.7
# 3649  1990-12-31  13.0
# 
# [3650 rows x 2 columns]

# =========================================================

# Line Plot
df.plot()


# Histogram Plot 
from matplotlib import pyplot 
df.hist()
pyplot.show()

# The data is following symmetric distribution(slightly positively skewed)


# Density Plot(Outline of Histogram Plot)
from matplotlib import pyplot 
df.plot(kind='kde')     # kde --> Kernal Density Estimation 
pyplot.show()


# ==============================================================


# Extract the year from Date column 
df['Year'] = pd.to_datetime(df['Date']).dt.year
df
#             Date  Temp  Year
# 0     1981-01-01  20.7  1981
# 1     1981-01-02  17.9  1981
# 2     1981-01-03  18.8  1981
# 3     1981-01-04  14.6  1981
# 4     1981-01-05  15.8  1981
#          ...   ...   ...
# 3645  1990-12-27  14.0  1990
# 3646  1990-12-28  13.6  1990
# 3647  1990-12-29  13.5  1990
# 3648  1990-12-30  15.7  1990
# 3649  1990-12-31  13.0  1990 

# [3650 rows x 3 columns]


# Create a Box Plot using Seaborn 
plt.figure(figsize=(12,8))
sns.boxplot(x='Year',y='Temp',data = df)
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.title('Year-wise Box plot of Temperature(1981-1990)')
plt.show() 


# ===============================================================

# Extract the month from the 'Date' Column 
df['Month'] = pd.to_datetime(df['Date']).dt.month
df
#             Date  Temp  Year  Month
# 0     1981-01-01  20.7  1981      1
# 1     1981-01-02  17.9  1981      1
# 2     1981-01-03  18.8  1981      1
# 3     1981-01-04  14.6  1981      1
# 4     1981-01-05  15.8  1981      1
#          ...   ...   ...    ...
# 3645  1990-12-27  14.0  1990     12
# 3646  1990-12-28  13.6  1990     12
# 3647  1990-12-29  13.5  1990     12
# 3648  1990-12-30  15.7  1990     12
# 3649  1990-12-31  13.0  1990     12
 
# [3650 rows x 4 columns]


# Create a Box Plot using seaborn 
plt.figure(figsize=(12,8))
sns.boxplot(x='Month',y='Temp',data=df)
plt.xlabel('Month')
plt.ylabel('Temperature')
plt.title('Month-wise Box plot of Temperature(1981-1990)')
plt.show()


# ==================================================================

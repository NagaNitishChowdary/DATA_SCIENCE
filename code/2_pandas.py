# -*- coding: utf-8 -*-
"""
Created on Fri May 24 00:24:58 2024

@author: Naga Nitish
"""

# pandas 

import numpy as np
import pandas as pd 

data = np.array(([24,170,75,100],[28,158,85,120],[30,165,70,115],[18,168,68,140],[34,168,65,125]))
df = pd.DataFrame(data)
df  # appers as table 

# assigning column names
names = ['Age','Height','Weight','BP']
df.columns = names
df

# assigning row names
df.index = ["A","B","C","D","E"]
df

# predefined functions 
df['Age'].mean()  # 26.8 
df['Age'].median() # 28.0
df['Age'].std()  # 6.09918
df['Age'].var() # 37.2
df['Age'].min() # 18
df['Age'].max() # 34


# opening csv file 
data = pd.read_csv("D:\\DS\\datasets\\nyc_weather.csv")
data

# printing first 5 rows
data.head()


# displaying last 5 rows
data.tail()


# displaying column names 
list(data)
# or 
data.columns


data['Temperature'].min()
data['Temperature'].mean()

# displaying whole information 
data['Temperature'].describe()

# when using 2/more variables use double brackets
data[['Temperature','WindSpeedMPH']]
data[data.columns[[1,6]]]


# to access series of columns 
#     rows cols
data.iloc[ : , 2:6] # all rows with 2 to 5 columns
data.iloc[0] #selects first row
data.iloc[0:2] # selects first 2 rows


# identify data types 
data.info()
data.dtypes


# counting the blanks ie null values 
data.isnull().sum()


# to display categories under certain specific column 
data['Events'].value_counts()


# in which days are raining 
data[data['Events'] == 'Rain']


# replacing null values with mean value
df = pd.read_csv("D:\\DS\\datasets\\first_file.csv")
df

# it doesn't save in database 
df['age'].fillna(value=df['age'].mean())

# it saves the mean value for null values in database 
df['age'].fillna(value=df['age'].mean(),inplace=True)
df

df1 = pd.read_csv("D:\\DS\\datasets\\second_file.csv")
df1


# combining 2 csv files 
# index will be 0,1,2,....0,1,2,....
df2 = pd.concat([df,df1])
df2

# index will be 0,1,2,3..........
# row concatenating (columns are same)
df2 = pd.concat([df,df1],ignore_index=True)
df2

# column concatenating (columns are different)
df3 = pd.read_csv('D:\\DS\\datasets\\third_file.csv')
df3

# axis = 1 tells that we need to do column concatenation
df4 = pd.concat([df2,df3],axis=1)
df4

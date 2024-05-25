# -*- coding: utf-8 -*-
"""
Created on Sat May 25 00:36:06 2024

@author: Naga Nitish
"""

import pandas as pd

df = pd.read_csv('D:\DS\datasets\market_3.csv')
df

df.shape


region = df['Region'].value_counts()
product = df['Product'].value_counts()


# bar plot ---> for discrete variables, it displays frequency
product.plot(kind='bar')
region.plot(kind='bar')



# scatter plot ---> for ranges (or) intervals
df['Stores'].hist()  
df['Stores'].skew() # it is positively skewed(left side data is more)
df['Stores'].kurt() # giving negative kurtosis(calculates peakedness)


# Construct the histogram for sales, inventory, returns and also 
# display skewness and kurtosis 

df.describe()
df['Sales'].hist() 
df['Sales'].skew()
df['Sales'].kurt()


df['Inventory'].hist()
df['Inventory'].skew()
df['Inventory'].kurt()

df['Returns'].hist()
df['Returns'].skew()
df['Returns'].kurt()


# box plot ---> to identify outliers
df.boxplot('Sales',vert=False) # gives horizontally
df.boxplot('Sales') # gives vertically 

df.boxplot('Inventory',vert=False)



# scatter plot ---> shows relationship between 2 continuous variables
import matplotlib.pyplot as plt 

# between sales and returns
plt.scatter(x=df['Sales'],y=df['Returns'])
plt.xlabel('Sales of the company')
plt.ylabel('Returns of the company')
plt.title('Sales vs Returns')
plt.show()

# between inventory vs sales 
plt.scatter(x=df['Inventory'],y=df['Sales'])
plt.xlabel('Inventory')
plt.ylabel('Sales of the company')
plt.title('Inventory vs Sales')
plt.show()

# Inventory vs return 
plt.scatter(x=df['Inventory'],y=df['Returns'])
plt.xlabel('Inventory')
plt.ylabel('Returns of the company')
plt.title('Inventory vs Returns')
plt.show()

# Correlation among these columns 
df[['Sales','Inventory','Returns','Inventory']].corr()
df[['Sales','Sales']].corr()  # correlation is 1 


# using seaborn
import seaborn as sns
sns.scatterplot(data = df,x='Sales',y='Inventory')
plt.plot()


# construct the boxplot for all the numeric variables and plot it  individually
# Get list of numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Loop through each numeric column and plot boxplot
for col in numeric_cols:
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
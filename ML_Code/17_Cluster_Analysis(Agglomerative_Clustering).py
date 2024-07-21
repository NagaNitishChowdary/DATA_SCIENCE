# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 16:47:17 2024

@author: Naga Nitish
"""

import numpy as np 
import pandas as pd 
 
df = pd.read_csv('17_shopping_data.csv')
df


df['Age'].describe()
# count    200.000000
# mean      38.850000
# std       13.969007
# min       18.000000
# 25%       28.750000
# 50%       36.000000
# 75%       49.000000
# max       70.000000
# Name: Age, dtype: float64


df['Annual Income (k$)'].describe()
# count    200.000000
# mean      60.560000
# std       26.264721
# min       15.000000
# 25%       41.500000
# 50%       61.500000
# 75%       78.000000
# max      137.000000
# Name: Annual Income (k$), dtype: float64


df['Spending Score (1-100)'].describe()
# count    200.000000
# mean      50.200000
# std       25.823522
# min        1.000000
# 25%       34.750000
# 50%       50.000000
# 75%       73.000000
# max       99.000000
# Name: Spending Score (1-100), dtype: float64



import matplotlib.pyplot as plt 
plt.scatter(x=df.iloc[:,3],y=df.iloc[:,4])
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.show()

X = df.iloc[:,3:]
X

# =====================================================================

# AGGLOMERATIVE CLUSTERING 

from sklearn.cluster import AgglomerativeClustering

# Single linkage method

cluster_single_linkage = AgglomerativeClustering(n_clusters=5,linkage='single') 
Y = cluster_single_linkage.fit_predict(X)

df['agglo_single_link'] = pd.DataFrame(Y)

df['agglo_single_link'].value_counts()
# 1    193
# 0      3
# 4      2
# 3      1
# 2      1
# Name: count, dtype: int64

# ---> not suitable 

# ======================================================================

# ---> Complete Linkage Method 

cluster_complete_linkage = AgglomerativeClustering(n_clusters=5,linkage='complete')
Y = cluster_complete_linkage.fit_predict(X)

df['agglo_complete_link'] = pd.DataFrame(Y)

df['agglo_complete_link'].value_counts()
# 1    85
# 0    39
# 2    32
# 4    23
# 3    21

# ---> suitable (check via scatter plot)

plt.figure(figsize=(10,7))
plt.scatter(X.iloc[:,0],X.iloc[:,1],c=cluster_complete_linkage.labels_,cmap='rainbow')
plt.show()


# ======================================================================

# ---> Average Linkage Method 

cluster_average_linkage = AgglomerativeClustering(n_clusters=5,linkage='average')
Y = cluster_average_linkage.fit_predict(X)

df['agglo_average_link'] = pd.DataFrame(Y)

df['agglo_average_link'].value_counts()
# 1    102
# 0     38
# 2     36
# 3     21
# 4      3

plt.figure(figsize=(10,7))
plt.scatter(X.iloc[:,0],X.iloc[:,1],c=cluster_average_linkage.labels_,cmap='rainbow')
plt.show()


# =======================================================================

# ---> Ward Linkage Method 

cluster_ward_linkage = AgglomerativeClustering(n_clusters=5,linkage='ward')
Y = cluster_ward_linkage.fit_predict(X)

df['agglo_ward_link'] = pd.DataFrame(Y)

df['agglo_ward_link'].value_counts()
# 1    85
# 2    39
# 0    32
# 4    23
# 3    21

plt.figure(figsize=(10,7))
plt.scatter(X.iloc[:,0],X.iloc[:,1],c=cluster_ward_linkage.labels_,cmap='rainbow')
plt.show()

# ========================================================================


# HOW THE CLUSTER NUMBER IS DECIDED WITH OUT GRAPH ? 
# WE CAN TAKE THE CHANCE OF ANY NUMBER BETWEEN 2 TO 10 , 
# MAXIMUM NUMBER CAN BE DECIDED ON DOMAIN KNOWLEDGE 

# WE HAVE A METHOD CALLED SILHOUTTE SCORE (-1 TO 1)
# -1 <--- BAD CLUSTER FORMATION 
# +1 <--- EXCELLENT CLUSTER FORMATION 

# FOR CLUSTER NUMBER, WE WILL CALCULATE THE SILHOUETTE SCORE 
# WHERE WE GET THE HIGHEST SILHOUETTE SCORE THAT IS BEST CLUSTER TO DECIDE

from sklearn.metrics import silhouette_score

score = []

for i in range(2,11):
    cluster = AgglomerativeClustering(n_clusters=i,linkage='ward')
    Y = cluster.fit_predict(X)
    score.append(silhouette_score(X,Y))
    
import matplotlib.pyplot as plt 
plt.scatter(x=range(2,11),y=score)
plt.xlabel('cluster numbers')
plt.xlabel('silhouette score')
plt.show()


# single --> clusters = 2   , maximum_score = 0.5 
# complete --> clusters = 5 , maximum_score = 0.55 
# average --> clusters = 7  , maximum_score = 0.54
# ward --> clusters = 5     , maximum_score = 0.55

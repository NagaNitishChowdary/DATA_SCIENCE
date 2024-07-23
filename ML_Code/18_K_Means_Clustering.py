# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 01:36:42 2024

@author: Naga Nitish
"""

import numpy as np 
import pandas as pd 
 
df = pd.read_csv('17_shopping_data.csv')
df


import matplotlib.pyplot as plt 
plt.scatter(x=df.iloc[:,3],y=df.iloc[:,4])
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.show()

X = df.iloc[:,3:]
X

# ===================================================================

from sklearn.cluster import KMeans 
# n_init ---> how many times the algorithm should run 
kmeans = KMeans(n_clusters=5,n_init=20)
kmeans.fit(X)

Y = kmeans.predict(X)
df['K_Means_Cluster'] = pd.DataFrame(Y)
df.head()

plt.figure(figsize=(10,7))
plt.scatter(X.iloc[:,0],X.iloc[:,1],c=Y,cmap='rainbow')
plt.show()

# ===================================================================

# FOR CLUSTER NUMBER, WE WILL CALCULATE THE SILHOUETTE SCORE 
# WHERE WE GET THE HIGHEST SILHOUETTE SCORE THAT IS BEST CLUSTER TO DECIDE

from sklearn.metrics import silhouette_score

scores = []
for i in range(2,11):
    cluster = KMeans(n_clusters=i,n_init=20)
    Y = cluster.fit_predict(X)
    scores.append(silhouette_score(X, Y))
    

import matplotlib.pyplot as plt 
plt.scatter(x=range(2,11),y=scores)
plt.xlabel("cluster numbers")
plt.ylabel("silhouette scores")
plt.show()


# for clusters = 5 , the silhouette score is high(0.55) 

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 00:25:57 2024

@author: Naga Nitish
"""

# ===================================================================

import pandas as pd 
import numpy as np 

df = pd.read_csv('17_shopping_data.csv')
df
"""
     CustomerID   Genre  Age  Annual Income (k$)  Spending Score (1-100)
0             1    Male   19                  15                      39
1             2    Male   21                  15                      81
2             3  Female   20                  16                       6
3             4  Female   23                  16                      77
4             5  Female   31                  17                      40
..          ...     ...  ...                 ...                     ...
195         196  Female   35                 120                      79
196         197  Female   45                 126                      28
197         198    Male   32                 126                      74
198         199    Male   32                 137                      18
199         200    Male   30                 137                      83

[200 rows x 5 columns]
"""

import matplotlib.pyplot as plt 
plt.scatter(x=df.iloc[:,3],y=df.iloc[:,4])
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.show()

X = df.iloc[:,3:]
X
"""
     Annual Income (k$)  Spending Score (1-100)
0                    15                      39
1                    15                      81
2                    16                       6
3                    16                      77
4                    17                      40
..                  ...                     ...
195                 120                      79
196                 126                      28
197                 126                      74
198                 137                      18
199                 137                      83

[200 rows x 2 columns]
"""

# ====================================================================

# DBSCAN 
# TO REMOVE THE OUTLIERS FROM THE DATA WE ARE CHOOSING DBSCAN 

X = df.iloc[:,2:5]   # HERE WE ARE ADDITIONALLY CONSIDERING AGE (OUR CHOICE WE CAN IGNORE ALSO)
X
"""
     Age  Annual Income (k$)  Spending Score (1-100)
0     19                  15                      39
1     21                  15                      81
2     20                  16                       6
3     23                  16                      77
4     31                  17                      40
..   ...                 ...                     ...
195   35                 120                      79
196   45                 126                      28
197   32                 126                      74
198   32                 137                      18
199   30                 137                      83

[200 rows x 3 columns]
"""

# PERFORM STANDARDIZATION
from sklearn.preprocessing import StandardScaler 
SS = StandardScaler()
SS_X = SS.fit_transform(X)
SS_X
"""
array([[-1.42456879, -1.73899919, -0.43480148],
       [-1.28103541, -1.73899919,  1.19570407],
       ........................................
       [-0.49160182,  2.91767117, -1.25005425],
       [-0.6351352 ,  2.91767117,  1.27334719]])
"""

# HOW CLUSTER FORMATION WILL BE FORMED HERE ? 
# EVERY DATA POINT WILL BE POINTED OUT IN 3 WAYS. 
# 1) CORE POINT 
# 2) BORDER POINT 
# 3) NOISE POINT 

# WE ARE USING 2 PARAMETERS TO DIVIDE INTO 3 POINTS. 
# a) EPSILON(RADIUS)
# b) MIN POINTS 

from sklearn.cluster import DBSCAN 
dbscan = DBSCAN(eps=0.75, min_samples=3)
dbscan.fit(SS_X)

# SYSTEM WILL TREAT NULL POINTS AS -1 , 
# CORE POINTS AND DATA POINTS AS 0 

df['DBSCAN_cluster'] = dbscan.labels_
df.head()
#    CustomerID   Genre  ...  Spending Score (1-100)  DBSCAN_cluster
# 0           1    Male  ...                      39              -1
# 1           2    Male  ...                      81               0
# 2           3  Female  ...                       6              -1
# 3           4  Female  ...                      77               0
# 4           5  Female  ...                      40               0

df['DBSCAN_cluster'].value_counts()
#  0    193    ---> core and border points 
# -1      7    ---> 7 noise points(outliers)


# OUTLIERS 
df[df['DBSCAN_cluster'] == -1]
#      CustomerID   Genre  ...  Spending Score (1-100)  DBSCAN_cluster
# 0             1    Male  ...                      39              -1
# 2             3  Female  ...                       6              -1
# 11           12  Female  ...                      99              -1
# 19           20  Female  ...                      98              -1
# 194         195  Female  ...                      16              -1
# 196         197  Female  ...                      28              -1
# 198         199    Male  ...                      18              -1


df_final = df[df['DBSCAN_cluster'] != -1]
df_final
"""
     CustomerID   Genre  ...  Spending Score (1-100)  DBSCAN_cluster
1             2    Male  ...                      81               0
3             4  Female  ...                      77               0
4             5  Female  ...                      40               0
5             6  Female  ...                      76               0
6             7  Female  ...                       6               0
..          ...     ...  ...                     ...             ...
192         193    Male  ...                       8               0
193         194  Female  ...                      91               0
195         196  Female  ...                      79               0
197         198    Male  ...                      74               0
199         200    Male  ...                      83               0

[193 rows x 6 columns]
"""

X = df_final.iloc[:,3:]
X
"""
     Annual Income (k$)  Spending Score (1-100)  DBSCAN_cluster
1                    15                      81               0
3                    16                      77               0
4                    17                      40               0
5                    17                      76               0
6                    18                       6               0
..                  ...                     ...             ...
192                 113                       8               0
193                 113                      91               0
195                 120                      79               0
197                 126                      74               0
199                 137                      83               0

[193 rows x 3 columns]
"""

# NOW THE OUTLIERS ARE REMOVED 
# THEN APPLY THE ALGORITHM YOU WANT (AGLOMERATIVE/KMEANS.....)

# BECAUSE HERE THE DATA IS NOT IN ORBITARY FORM SO YOU CAN APPLY OTHER METHODS

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5,n_init=20)  # AS WE ALREADY CALCULATED THAT 5 SHOWS BEST RESULTS IN KMEANS FILE 
kmeans.fit(X)

Y = kmeans.predict(X)
df_final['Kmeans_Cluster'] = pd.DataFrame(Y)
df_final.head()
#   CustomerID   Genre  ...  DBSCAN_cluster  Kmeans_Cluster
#1           2    Male  ...               0             3.0
#3           4  Female  ...               0             3.0
#4           5  Female  ...               0             4.0
#5           6  Female  ...               0             3.0
#6           7  Female  ...               0             4.0


df_final['Kmeans_Cluster'].value_counts()
"""
Kmeans_Cluster
2.0    78
4.0    39
1.0    35
3.0    19
0.0    18
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 21:35:45 2024

@author: Naga Nitish
"""

import numpy as np 
import pandas as pd 

df = pd.read_csv('22_Movie.csv')
df

df.shape
# (8992, 3)

df.head()
#    userId             movie  rating
# 0       3  Toy Story (1995)     4.0
# 1       6  Toy Story (1995)     5.0
# 2       8  Toy Story (1995)     4.0
# 3      10  Toy Story (1995)     4.0
# 4      11  Toy Story (1995)     4.5


# Number of unique users
len(df.userId.unique())
# 4081


# Number of movies
len(df.movie.unique())


df.movie.value_counts()
# movie
# Toy Story (1995)                      2569
# GoldenEye (1995)                      1548
# Heat (1995)                           1260
# Jumanji (1995)                        1155
# Sabrina (1995)                         700
# Grumpier Old Men (1995)                685
# Father of the Bride Part II (1995)     657
# Sudden Death (1995)                    202
# Waiting to Exhale (1995)               138
# Tom and Huck (1995)                     78

# =================================================================

user_df = df.pivot(index='userId',columns='movie',values='rating')
user_df

# movie   Father of the Bride Part II (1995)  ...  Waiting to Exhale (1995)
# userId                                      ...                          
# 1                                      NaN  ...                       NaN
# 2                                      NaN  ...                       NaN
# ..                                     ...  ...                       ...
# [4081 rows x 10 columns]


# Replacing those NaN values with 0's 

user_df.fillna(0,inplace=True)
user_df 

# movie   Father of the Bride Part II (1995)  ...  Waiting to Exhale (1995)
# userId                                      ...                          
# 1                                      0.0  ...                       0.0
# 2                                      0.0  ...                       0.0
# 3                                      0.0  ...                       0.0
# 4                                      0.0  ...                       0.0
# 5                                      0.0  ...                       0.0
#                                    ...  ...                       ...
# 7115                                   4.0  ...                       0.0
# 7116                                   3.5  ...                       0.0
# 7117                                   0.0  ...                       0.0
# 7119                                   0.0  ...                       0.0
# 7120                                   0.0  ...                       0.0
# [4081 rows x 10 columns]

# ===============================================================


# Calculating Cosine Similarity between users 
from sklearn.metrics import pairwise_distances 
user_simi = 1 - pairwise_distances(user_df,metric='cosine')
user_simi 
# array([[1.        , 0.        , 0.        , ..., 0.        , 0.        ,
#        0.55337157],
#       [0.        , 1.        , 0.        , ..., 0.45883147, 0.        ,
#        0.        ],
#       [0.        , 0.        , 1.        , ..., 0.45883147, 1.        ,
#        0.62254302],
#       ...,
#       [0.        , 0.45883147, 0.45883147, ..., 1.        , 0.45883147,
#        0.47607054],
#       [0.        , 0.        , 1.        , ..., 0.45883147, 1.        ,
#        0.62254302],
#       [0.55337157, 0.        , 0.62254302, ..., 0.47607054, 0.62254302,
#        1.        ]])

user_simi.shape
# (4081, 4081)


# Fill the diagonal values as 0 
np.fill_diagonal(user_simi, 0)


user_simi_df = pd.DataFrame(user_simi)
user_simi_df

#           0         1         2     ...      4078      4079      4080
# 0     0.000000  0.000000  0.000000  ...  0.000000  0.000000  0.553372
# 1     0.000000  0.000000  0.000000  ...  0.458831  0.000000  0.000000
# 2     0.000000  0.000000  0.000000  ...  0.458831  1.000000  0.622543
# 3     0.000000  0.000000  0.000000  ...  0.619422  0.000000  0.000000
# 4     1.000000  0.000000  0.000000  ...  0.000000  0.000000  0.553372
#        ...       ...       ...  ...       ...       ...       ...
# 4076  0.000000  0.000000  0.000000  ...  0.000000  0.000000  0.000000
# 4077  0.000000  0.000000  0.752577  ...  0.345306  0.752577  0.468511
# 4078  0.000000  0.458831  0.458831  ...  0.000000  0.458831  0.476071
# 4079  0.000000  0.000000  1.000000  ...  0.458831  0.000000  0.622543
# 4080  0.553372  0.000000  0.622543  ...  0.476071  0.622543  0.000000 
 
# [4081 rows x 4081 columns]

# ===================================================================


# Set the index names and column names as UserId's 

user_simi_df.index = df.userId.unique() 
user_simi_df.columns = df.userId.unique() 


user_simi_df
#           3         6         8     ...      7080      7087      7105
# 3     0.000000  0.000000  0.000000  ...  0.000000  0.000000  0.553372
# 6     0.000000  0.000000  0.000000  ...  0.458831  0.000000  0.000000
# 8     0.000000  0.000000  0.000000  ...  0.458831  1.000000  0.622543
# 10    0.000000  0.000000  0.000000  ...  0.619422  0.000000  0.000000
# 11    1.000000  0.000000  0.000000  ...  0.000000  0.000000  0.553372
#        ...       ...       ...  ...       ...       ...       ...
# 7044  0.000000  0.000000  0.000000  ...  0.000000  0.000000  0.000000
# 7070  0.000000  0.000000  0.752577  ...  0.345306  0.752577  0.468511
# 7080  0.000000  0.458831  0.458831  ...  0.000000  0.458831  0.476071
# 7087  0.000000  0.000000  1.000000  ...  0.458831  0.000000  0.622543
# 7105  0.553372  0.000000  0.622543  ...  0.476071  0.622543  0.000000

#[4081 rows x 4081 columns]


# ===================================================================

# Now we have to find the maximum value from every row 

user_simi_df.idxmax(axis=1)[0:user_simi_df.shape[0]]
# 3         11
# 6        168
# 8         16
# 10      4047
# 11         3
# ...
# 7044      80
# 7070    1808
# 7080     708
# 7087       8
# 7105    4110
# Length: 4081

df[df['userId'] == 3]
#    userId             movie  rating
# 0       3  Toy Story (1995)     4.0

df[df['userId'] == 11]
#       userId             movie  rating
# 4         11  Toy Story (1995)     4.5
# 7446      11  GoldenEye (1995)     2.5

# *** We can recommend GoldenEye movie to userId 3 


df[(df['userId'] == 6) | (df['userId'] == 168)]
#       userId                    movie  rating
# 1          6         Toy Story (1995)     5.0
# 60       168         Toy Story (1995)     4.5
# 3725       6  Grumpier Old Men (1995)     3.0
# 6464       6           Sabrina (1995)     5.0

# *** We can recommend Grumpier Old Men and Sabrica to userId 168 

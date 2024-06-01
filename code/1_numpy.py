# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Naga Nitish
"""

# numpy 

import numpy as np 

# creating an numpy array 
a = np.array(([24,170],[28,158],[30,165],[18,168],[34,168]))
a

# ndim attribute gives number of dimensions 
a.ndim  # 2

# shape attribute returns a tuple representing shape of array 
a.shape # (5 , 2)

# reshape method used to change the shape of an array 
a.reshape(2,5)

# type method used to determine type of object 
type(a)  # numpy.ndarray

# arange method 
np.arange(5)  # 0,1,2,3,4
np.arange(2,6) # 2,3,4,5
np.arange(2,10,2) # 2,4,6,8

# around method rounds the elements to a certain number 
a2 = [1.3467,3.10987,4.912]
np.around(a2,1)  # array([1.3,3.1,4.9])

# sqrt method computes the square root
np.sqrt(5)  # 2.23606

# astype method used to cast an array into specific datatype
a2 =np.array(a2)
a2 = a2.astype(int)
a2  # [1,3,4] 

# axis = 0 => row-wise 
# axis = 1 => column-wise
np.sum(a,0) # [134, 829]
np.sum(a,1) # [194, 186, 195, 186, 202]

np.mean(a,0) # [26.8, 165.8]
np.mean(a,1) # [97.0, 93.0, 97.5, 93.0, 101.0]


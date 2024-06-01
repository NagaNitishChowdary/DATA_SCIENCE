# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 09:50:38 2024

@author: Naga Nitish
"""

# table valus of z 

import scipy.stats as stats

# For any one sided alpha = 5% 
stats.norm.ppf(.95).round(3)   # 1.645

# For any one sided alpha = 10%
stats.norm.ppf(.90).round(3)   # 1.282




# table value of f 

import scipy.stats as stats

# dfn = k-1 
# dfd = n-k 
stats.f(dfn=2,dfd=33).ppf(0.95).round(3)    # 3.285
# k = how many groups u are comparing
# n = number of samples
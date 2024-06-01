# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 20:26:38 2024

@author: Naga Nitish
"""

import numpy as np

# sample size 1 
n1 = 247

# sample size 2
n2 = 308 

alpha = 0.05

p1 = 37 # from state 1, 37% students report that they got job immediately after graduation
p2 = 39 # from state 2, 39% students reported

props = np.array([p1,p2])
sampsize = np.array([n1,n2])

from statsmodels.stats.proportion import proportions_ztest
stat, pval = proportions_ztest(props,sampsize)

pval  # 0.429980

if pval < alpha :
    print('H0 is rejected and H1 is accepted')
else:
    print('H0 is accepted and H1 is rejected')
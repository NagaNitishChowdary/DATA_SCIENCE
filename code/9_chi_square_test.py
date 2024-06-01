# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 21:14:38 2024

@author: Naga Nitish
"""

# chi square test of independence 

import pandas as pd

df = pd.read_csv('9_credit_new.csv')
df

df['Cards']
df['Ethnicity']

!pip install researchpy
import researchpy as rp

table =rp.crosstab(df['Cards'],df['Ethnicity'],test='chi-square')
table

# output
#                 Ethnicity                     
# Ethnicity African American Asian Caucasian  All
# Cards                                          
# 1                        6    13        32   51
# 2                       34    31        50  115
# 3                       34    26        51  111
# 4                       11    16        45   72
# 5                       14    16        21   51
# All                     99   102       199  400,
#                 Chi-square test  results
# 0  Pearson Chi-square ( 8.0) =   16.2092
# 1                    p-value =    0.0395
# 2                 Cramer's V =    0.1423)

# results 
pvalue = 0.0395
alpha = 0.05

if pvalue < alpha:
    print('H0 is rejected and H1 is accepted')
else:
    print('H0 is accepted and H1 is rejected')
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 22:55:27 2024

@author: Naga Nitish
"""

# TWO SAMPLE Z-TEST

import pandas as pd 
df = pd.read_csv('6_Cars_100.csv')
df


df['USCARS'].mean()   # 14.828
df['GERMANCARS'].mean()  # 16.5029


from scipy.stats import ttest_ind
zcal , pvalue = ttest_ind(df['USCARS'],df['GERMANCARS'])
zcal    # -3.02

# If pval is lesser than alpha(5% = 0.05), then 
# H0 is rejected, H1 is accepted 

# If pval is greater than alpha(5% = 0.05), then 
# H0 is accepted, H1 is rejected 

# pvalue always lies between 0 to 1 
# most of the people using p value as final instead of z calculated value, 
# since zcal value falls anywhere between -inf to +inf but our p-values 
# will be converts z value in between 0 to 1 only.

alpha = 0.05
if(pvalue < alpha):
    print('H0 is rejected and H1 is accepted')
else:
    print('H0 is accepted and H1 is rejected')
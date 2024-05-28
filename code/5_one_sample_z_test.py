# -*- coding: utf-8 -*-
"""
Created on Tue May 28 08:24:50 2024

@author: Naga Nitish
"""

import pandas as pd 
df = pd.read_csv('LungCapdata.csv')
df

df['LungCap'].mean()

# one sample z test 
from statsmodels.stats import weightstats
Zcal, Pval = weightstats.ztest(df['LungCap'],value=8,alternative='smaller')
print("Z calculated value: ",Zcal.round(3))  # -1.3 which is right of -1.6
# so, it falls under accepted region 



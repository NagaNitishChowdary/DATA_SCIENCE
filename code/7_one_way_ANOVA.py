# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 11:09:39 2024

@author: Naga Nitish
"""

import pandas as pd 

df = pd.read_csv('7_Dietplan.csv')
df


from statsmodels.formula.api import ols 
anovaVal = ols('calories ~ C(Dietplans)',data=df).fit() 
anovaVal  # <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x1ddb0abd990>


import statsmodels.api as sm 
table = sm.stats.anova_lm(anovaVal,type=1)  # type = 1 -> one way anova 
table

#                 df      sum_sq    mean_sq         F    PR(>F)
# C(Dietplans)   2.0   38.888889  19.444444  3.571429  0.039441
# Residual      33.0  179.666667   5.444444       NaN       NaN


pvalue = 0.03944
alpha = 0.05

if pvalue < alpha:
    print('Ho is rejected and H1 is accepted')
else:
    print('H0 is accepted and H1 is rejected')
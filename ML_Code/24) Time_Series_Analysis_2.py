# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 12:21:41 2024

@author: Naga Nitish
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_csv('24_footfalls.csv')
df

#       Month  Footfalls    t  log_footfalls  t_square  ...  Aug  Sep  Oct  Nov  Dec
# 0    Jan-91       1709    1       7.443664         1  ...    0    0    0    0    0
# 1    Feb-91       1621    2       7.390799         4  ...    0    0    0    0    0
# 2    Mar-91       1973    3       7.587311         9  ...    0    0    0    0    0
# 3    Apr-91       1812    4       7.502186        16  ...    0    0    0    0    0
# 4    May-91       1975    5       7.588324        25  ...    0    0    0    0    0
# ..      ...        ...  ...            ...       ...  ...  ...  ...  ...  ...  ...
# 154  Nov-03       2076  155       7.638198     24025  ...    0    0    0    1    0
# 155  Dec-03       2141  156       7.669028     24336  ...    0    0    0    0    1
# 156  Jan-04       1832  157       7.513164     24649  ...    0    0    0    0    0
# 157  Feb-04       1838  158       7.516433     24964  ...    0    0    0    0    0
# 158  Mar-04       2132  159       7.664816     25281  ...    0    0    0    0    0

# [159 rows x 17 columns]

# =======================================================================


# Line Plot 
df['Footfalls'].plot()

# =====================================================================

df['Date'] = pd.to_datetime(df.Month,format="%b-%y")
df.head()
#    Month  Footfalls  t  log_footfalls  t_square  ...  Sep  Oct  Nov  Dec       Date
# 0  Jan-91       1709  1       7.443664         1  ...    0    0    0    0 1991-01-01
# 1  Feb-91       1621  2       7.390799         4  ...    0    0    0    0 1991-02-01
# 2  Mar-91       1973  3       7.587311         9  ...    0    0    0    0 1991-03-01
# 3  Apr-91       1812  4       7.502186        16  ...    0    0    0    0 1991-04-01
# 4  May-91       1975  5       7.588324        25  ...    0    0    0    0 1991-05-01


df['month'] = df.Date.dt.strftime("%b")  # Month Extraction 
df['year'] = df.Date.dt.strftime("%Y")  # Year Extraction 
df.head()
#     Month  Footfalls  t  log_footfalls  ...  Dec       Date  month  year
# 0  Jan-91       1709  1       7.443664  ...    0 1991-01-01    Jan  1991
# 1  Feb-91       1621  2       7.390799  ...    0 1991-02-01    Feb  1991
# 2  Mar-91       1973  3       7.587311  ...    0 1991-03-01    Mar  1991
# 3  Apr-91       1812  4       7.502186  ...    0 1991-04-01    Apr  1991
# 4  May-91       1975  5       7.588324  ...    0 1991-05-01    May  199

# =======================================================================

heatmap_y_month = pd.pivot_table(data=df,values='Footfalls',index='year',columns='month',fill_value=0)
heatmap_y_month
# month     Apr     Aug     Dec     Feb  ...     May     Nov     Oct     Sep
# year                                   ...                                
# 1991   1812.0  2013.0  1814.0  1621.0  ...  1975.0  1676.0  1725.0  1596.0
# 1992   1956.0  1997.0  1875.0  1557.0  ...  1885.0  1862.0  1810.0  1704.0
# 1993   1957.0  1996.0  1734.0  1619.0  ...  1917.0  1720.0  1753.0  1673.0
# 1994   1834.0  1907.0  1783.0  1574.0  ...  1831.0  1776.0  1779.0  1686.0
# 1995   1733.0  1875.0  1657.0  1497.0  ...  1772.0  1673.0  1647.0  1571.0
# 1996   1608.0  1943.0  1700.0  1361.0  ...  1697.0  1576.0  1687.0  1551.0
# 1997   1655.0  2008.0  1797.0  1372.0  ...  1763.0  1732.0  1774.0  1616.0
# 1998   1825.0  1922.0  1847.0  1413.0  ...  1843.0  1817.0  1791.0  1670.0
# 1999   1840.0  1949.0  1836.0  1549.0  ...  1846.0  1850.0  1804.0  1607.0
# 2000   1971.0  2097.0  2000.0  1617.0  ...  1992.0  1981.0  1977.0  1824.0
# 2001   2024.0  2203.0  1985.0  1663.0  ...  2047.0  1974.0  1951.0  1708.0
# 2002   2048.0  2027.0  1996.0  1771.0  ...  2069.0  1858.0  1917.0  1734.0
# 2003   2099.0  2174.0  2141.0  1749.0  ...  2105.0  2076.0  2121.0  1931.0
# 2004      0.0     0.0     0.0  1838.0  ...     0.0     0.0     0.0     0.0


import matplotlib.pyplot as plt 
import seaborn as sns 
plt.figure(figsize=(10,6))
sns.heatmap(heatmap_y_month,annot=True,fmt='g')  # fmt is format of the grid value

# =====================================================================

# Box plot for every month 

plt.figure(figsize=(8,6))
sns.boxplot(x='month',y='Footfalls',data=df)


# Box plot for every year 

plt.figure(figsize=(8,6))
sns.boxplot(x='year',y='Footfalls',data=df)

# ======================================================================

# SPLITING THE DATA 

df.shape # (159, 20)
Train = df.head(147)
Test = df.tail(12)
Test
#       Month  Footfalls    t  log_footfalls  ...  Dec       Date  month  year
# 147  Apr-03       2099  148       7.649216  ...    0 2003-04-01    Apr  2003
# 148  May-03       2105  149       7.652071  ...    0 2003-05-01    May  2003
# 149  Jun-03       2130  150       7.663877  ...    0 2003-06-01    Jun  2003
# 150  Jul-03       2223  151       7.706613  ...    0 2003-07-01    Jul  2003
# 151  Aug-03       2174  152       7.684324  ...    0 2003-08-01    Aug  2003
# 152  Sep-03       1931  153       7.565793  ...    0 2003-09-01    Sep  2003
# 153  Oct-03       2121  154       7.659643  ...    0 2003-10-01    Oct  2003
# 154  Nov-03       2076  155       7.638198  ...    0 2003-11-01    Nov  2003
# 155  Dec-03       2141  156       7.669028  ...    1 2003-12-01    Dec  2003
# 156  Jan-04       1832  157       7.513164  ...    0 2004-01-01    Jan  2004
# 157  Feb-04       1838  158       7.516433  ...    0 2004-02-01    Feb  2004
# 158  Mar-04       2132  159       7.664816  ...    0 2004-03-01    Mar  2004

# [12 rows x 20 columns]

# ======================================================================

# Now Applying Time Series Models 

import statsmodels.formula.api as smf 

# LINEAR MODEL  ---> Yt = B0(Beta not) + B1t(Beta1 t) + e(epsilon)
linear_model = smf.ols('Footfalls~t',data=Train).fit()

# OLS stands are "Ordinary Least Squares". It is a method used for estimating
# the unknown parameters in a linear regression model. 

# The goal of OLS is to find the best fitting line through the data points 
# by minimizing the sum of the squared differences between the oberserved values
# and the values predicted by the linear model.

# Here only t is the input ---> training data is sending inside 

pred_linear = pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))

rmse_linear = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_linear))**2))
rmse_linear
# 209.92559265462594


# =====================================================================

# Exponential Model  ---> Log(Yt) = B0(Beta not) + B1t(Beta1 t) + e(epsilon)

# Here same as linear model, but Y variable need to be changed. 

# Training
Exp = smf.ols('log_footfalls~t',data=Train).fit()

# Prediction 
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))

# Error
rmse_Exp = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(np.exp(pred_Exp)))**2))
rmse_Exp
# 217.05263569548086

# =====================================================================

# QUADRATIC MODEL ---> (Yt) = B0(Beta not) + B1t(Beta1 t) + B2t2(Beta2 tSquare) + e(epsilon)
# Here t and t2(t square) are the inputs 

# Training
Quad = smf.ols('Footfalls~t+t_square',data=Train).fit()

# Testing
pred_Quad = pd.Series(Quad.predict(pd.DataFrame(Test[['t','t_square']])))

# Error 
rmse_Quad = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_Quad))**2))
rmse_Quad
# 137.1546274135642

# ======================================================================

# ADDITIVE SEASONILITY ---> (Yt) = B0(Beta not) + B1Djan(Beta1 Djan) + B2Dfeb(Beta2 Dfeb) + ... + B11Dnov(Beta11 Dnov) + e(epsilon)

# Training
add_sea = smf.ols('Footfalls~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Train).fit() 

# Testing
pred_add_sea = pd.Series(add_sea.predict(pd.DataFrame(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']])))

# Error
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_add_sea))**2))
rmse_add_sea
# 264.66439005687863

# =====================================================================

# ADDITIVE SEASONALITY QUADRATIC (3rd Model + 4th Model)
# (Yt) = B0(Beta not) + B1t(Beta1 t) + B2t2(Beta2 tSquare) + B3Djan(Beta3 Djan) + B4Dfeb(Beta4 Dfeb) + ... + B11Dnov(Beta11 Dnov) + e(epsilon)

# Training
add_sea_Quad = smf.ols('Footfalls~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Train).fit()

# Testing 
pred_add_sea_Quad = pd.Series(add_sea_Quad.predict(pd.DataFrame(Test[['t','t_square','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']])))

# Error 
rmse_add_sea_Quad = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_add_sea_Quad))**2))
rmse_add_sea_Quad
# 50.60724584153715   ----> better than above 4 models 


# =======================================================================

# MULTIPLICATIVE SEASONLITY 
# (Log Yt) = B0(Beta not) + B1Djan(Beta1 Djan) + B2Dfeb(Beta2 Dfeb) + ... + B11Dnov(Beta11 Dnov) + e(epsilon)

# Training
mul_sea = smf.ols('log_footfalls~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Train).fit()

# Testing 
pred_mul_sea = pd.Series(mul_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))

# Error
rmse_mul_sea = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(np.exp(pred_mul_sea)))**2))
rmse_mul_sea
# 268.1970325266327

# =========================================================================

# Multiplicative Additive Seasonality 

# Training
Mul_Add_sea = smf.ols('log_footfalls~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Train).fit()

# Testing
pred_Mul_Add_sea= pd.Series(Mul_Add_sea.predict(Test[['t','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))

rmse_Mul_Add_sea = np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(np.exp(pred_Mul_Add_sea)))**2))
rmse_Mul_Add_sea
# 172.76726787495969

# ======================================================================


# COMPARING ALL THE RESULTS 


data = {"MODEL" : pd.Series(["rmse_linear", "rmse_Exp", "rmse_Quad", "rmse_add_sea", "rmse_add_sea_Quad", "rmse_mul_sea", "rmse_Mul_Add_sea"]), 
        "RMSE_Values": pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_Quad,rmse_mul_sea,rmse_Mul_Add_sea])}
data
# {'MODEL':                  'RMSE_Values': 
# 0          rmse_linear      0    209.925593
# 1             rmse_Exp      1    217.052636
# 2            rmse_Quad      2    137.154627
# 3         rmse_add_sea      3    264.664390
# 4    rmse_add_sea_Quad      4     50.607246
# 5         rmse_mul_sea      5    268.197033
# 6     rmse_Mul_Add_sea      6    172.767268
# dtype: object,              dtype: float64}


table_rmse = pd.DataFrame(data)
table_rmse
#               MODEL  RMSE_Values
# 0        rmse_linear   209.925593
# 1           rmse_Exp   217.052636
# 2          rmse_Quad   137.154627
# 3       rmse_add_sea   264.664390
# 4  rmse_add_sea_Quad    50.607246
# 5       rmse_mul_sea   268.197033
# 6   rmse_Mul_Add_sea   172.767268
 
table_rmse.sort_values(['RMSE_Values'])
#                MODEL  RMSE_Values
# 4  rmse_add_sea_Quad    50.607246
# 2          rmse_Quad   137.154627
# 6   rmse_Mul_Add_sea   172.767268
# 0        rmse_linear   209.925593
# 1           rmse_Exp   217.052636
# 3       rmse_add_sea   264.664390
# 5       rmse_mul_sea   268.197033

# =======================================================================

# NOW LETS TRAIN ALL THE DATA AND TRY TO PREDICT THE FUTURE 

# BUILD THE MODEL ON THE ENTIRE DATA SET 

model_full = smf.ols('Footfalls~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=df).fit()


new_data = pd.read_csv('24_Predict_new.csv')
new_data


# Predicting on future data 
pred_new = pd.Series(model_full.predict(new_data))
pred_new
# 0     2193.807626
# 1     2229.969736
# 2     2200.670308
# 3     2311.293957
# 4     2356.071452
# 5     2036.848947
# 6     2187.241826
# 7     2181.480859
# 8     2234.104508
# 9     1999.997498
# 10    1972.995363
# 11    2280.493228
# 12            NaN
# dtype: float64


new_data['forecasted_Footfalls'] = pd.Series(pred_new)
new_data
#      Month  Jan  Feb  Mar  Apr  ...  Nov  Dec      t  t_square  forecasted_Footfalls
# 0   01-Apr  0.0  0.0  0.0  1.0  ...  0.0  0.0  160.0   25600.0           2193.807626
# 1   01-May  0.0  0.0  0.0  0.0  ...  0.0  0.0  161.0   25921.0           2229.969736
# 2   01-Jun  0.0  0.0  0.0  0.0  ...  0.0  0.0  162.0   26244.0           2200.670308
# 3   01-Jul  0.0  0.0  0.0  0.0  ...  0.0  0.0  163.0   26569.0           2311.293957
# 4   01-Aug  0.0  0.0  0.0  0.0  ...  0.0  0.0  164.0   26896.0           2356.071452
# 5   01-Sep  0.0  0.0  0.0  0.0  ...  0.0  0.0  165.0   27225.0           2036.848947
# 6   01-Oct  0.0  0.0  0.0  0.0  ...  0.0  0.0  166.0   27556.0           2187.241826
# 7   01-Nov  0.0  0.0  0.0  0.0  ...  1.0  0.0  167.0   27889.0           2181.480859
# 8   01-Dec  0.0  0.0  0.0  0.0  ...  0.0  1.0  168.0   28224.0           2234.104508
# 9   01-Jan  1.0  0.0  0.0  0.0  ...  0.0  0.0  169.0   28561.0           1999.997498
# 10  01-Feb  0.0  1.0  0.0  0.0  ...  0.0  0.0  170.0   28900.0           1972.995363
# 11  01-Mar  0.0  0.0  1.0  0.0  ...  0.0  0.0  171.0   29241.0           2280.493228
# [12 rows x 16 columns]


# ========================================================================

# CONCATENATE THE PREDICTED DATA AND ORIGINAL DATA 

final_data = pd.concat([df,new_data])
final_data 


# Constructing the graph for current data and future data 

final_data[['Footfalls','forecasted_Footfalls']].reset_index(drop=True).plot()

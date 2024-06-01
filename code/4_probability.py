# -*- coding: utf-8 -*-
"""
Created on Sat May 25 23:10:50 2024

@author: Naga Nitish
"""

# scipy ---> Scientific Python 


# BINOMIAL DISTRIBUTION (DISCRETE DATA)


from scipy.stats import binom 

# Example 1 
# 5 people applied for home loan, as per prev. records only 60% eligible,
# what is the probability of issuing home loan "exactly" for 3 people 

bi = binom(n = 5,p = 0.6)  # n,p

# P(X == 3)
x1 = bi.pmf(3)  # probability mass function used because of discrete data 
x1.round(3)

# Probability of "exactly" 2 people
x2 = bi.pmf(2)
x2.round(3)

# what is the probability of issuing home loan for atmost 2 people 
# p[X <= 2] = p[X = 0] + p[X = 1] + p[X = 2]
x3 = bi.cdf(2)  # Cumulative distributive function 
x3.round(3)

# what is the probability of issuing home loan for atleast 3 people 
# p[X >= 3] = p[X = 3] + p[X = 4] + p[X = 5]
x4 = 1 - bi.cdf(2)
x4.round(3)




# Example 2 
# 80% of all business start ups in the IT industry report that they generate
# a profit in their first year. If a sample of 10 new IT bussiness 
# startups is selected, find the probability that "exactly" seven will 
# generate a profit in their first year. 

# n = 10 , p = 0.8 , q = 0.2 
eg2 = binom(n=10, p=0.8)
x5 = eg2.pmf(7) # exactly 7  
x5.round(3)



# Example 3 
# 70% of adults who are with corona positive reports giving the feedback  
# that they got a relief with a specific medication. 

# If the same medication given to another 250 patients, what is the 
# probability the medication will effect atleast 160 patients 

# p = 0.7 , n = 250 
eg3 = binom(n=250, p=0.7)
# p[X >= 160]
x6 = 1 - eg3.cdf(159)
x6.round(3)

# ==========================================================================

# Normal Distribution 

# Proving using python 

from scipy.stats import norm 

nd = norm(170,10) # mean, standard deviation 

# P(X <= 170)
x11 = nd.cdf(170)
x11.round(3)  # 50 ===> 50% data is present before mean


# P(X >= 170)
x22 = 1 - nd.cdf(170)
x22.round(3)  # 50


# P(160 < X < 180)
x33 = nd.cdf(180) - nd.cdf(160)
x33.round(3)  # 68.3

# P(150 < X < 190)
x44 = nd.cdf(190) - nd.cdf(150)
x44.round(3)  # 95.4

# P(140 < X < 200)
x55 = nd.cdf(200) - nd.cdf(140)
x55.round(3) # 99.7%

# P(X < 150)
x66 = nd.cdf(150)
x66.round(3)  # 2.3%

# P(165 < X < 185)
x77 = nd.cdf(185) - nd.cdf(165)
x77.round(3)



# Example 2 

# A radar unit is used to measure speeds of cars on a highway. The speeds are normally
# distributed with a mean of 90 kmph and a standard deviation of 10 kmph.
# What is the probability that a car picked at random is travelling at more than 
# 100 kmph

# from scipy.stats import norm 
nd2 = norm(90,10) # mean, sd 
nd2

# P(X > 100)
x111 = 1 - nd2.cdf(100)
x111.round(3)  
# There are 15.9% chances are there that any vehicles are travelling more than 100



# Example 3 

# For a certain type of mobiles, the length of time between charges of the battery 
# is normally distributed with a mean of 10 hours and a standard deviation 20 minutes. 
# John owns one of these mobiles and wants to the probability that the length of line 
# will be between 10 and 11 hours. 

nd3 = norm(600,20) # mean,sd
nd3

x1111 = nd3.cdf(660) - nd3.cdf(600)
x1111.round(3)










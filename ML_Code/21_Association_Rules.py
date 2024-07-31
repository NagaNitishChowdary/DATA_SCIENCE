# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 06:57:26 2024

@author: Naga Nitish
"""

# pip install apyori

import numpy as np 
import pandas as pd 
 
df = pd.read_csv('21_Market_Basket_Optimisation.csv',header=None)
# here header=None because no variable names for this dataset 
df.shape
# (7501, 20)


trans = [] 
for i in range(0,df.shape[0]):
    trans.append([str(df.values[i,j]) for j in range(0,20)])
    
# trans[0]  ---> first person purchased items 
# len(trans)  ---> 7501

from apyori import apriori
# apriori expects the input data as the list format 

rules = apriori(transactions=trans, 
                min_support = 0.003,
                min_confidence = 0.2,
                min_lift = 3,
                min_length = 2, 
                max_length = 2)

rules = list(rules)
rules
#[RelationRecord(items=frozenset({'light cream', 'chicken'}), support=0.004532728969470737, ordered_statistics=[OrderedStatistic(items_base=frozenset({'light cream'}), items_add=frozenset({'chicken'}), confidence=0.29059829059829057, lift=4.84395061728395)]),
# RelationRecord(items=frozenset({'mushroom cream sauce', 'escalope'}), support=0.005732568990801226, ordered_statistics=[OrderedStatistic(items_base=frozenset({'mushroom cream sauce'}), items_add=frozenset({'escalope'}), confidence=0.3006993006993007, lift=3.790832696715049)]),
# RelationRecord(items=frozenset({'pasta', 'escalope'}), support=0.005865884548726837, ordered_statistics=[OrderedStatistic(items_base=frozenset({'pasta'}), items_add=frozenset({'escalope'}), confidence=0.3728813559322034, lift=4.700811850163794)]),
# RelationRecord(items=frozenset({'fromage blanc', 'honey'}), support=0.003332888948140248, ordered_statistics=[OrderedStatistic(items_base=frozenset({'fromage blanc'}), items_add=frozenset({'honey'}), confidence=0.2450980392156863, lift=5.164270764485569)]),
# RelationRecord(items=frozenset({'herb & pepper', 'ground beef'}), support=0.015997866951073192, ordered_statistics=[OrderedStatistic(items_base=frozenset({'herb & pepper'}), items_add=frozenset({'ground beef'}), confidence=0.3234501347708895, lift=3.2919938411349285)]),
# RelationRecord(items=frozenset({'tomato sauce', 'ground beef'}), support=0.005332622317024397, ordered_statistics=[OrderedStatistic(items_base=frozenset({'tomato sauce'}), items_add=frozenset({'ground beef'}), confidence=0.3773584905660377, lift=3.840659481324083)]),
# RelationRecord(items=frozenset({'light cream', 'olive oil'}), support=0.003199573390214638, ordered_statistics=[OrderedStatistic(items_base=frozenset({'light cream'}), items_add=frozenset({'olive oil'}), confidence=0.20512820512820515, lift=3.1147098515519573)]),
# RelationRecord(items=frozenset({'whole wheat pasta', 'olive oil'}), support=0.007998933475536596, ordered_statistics=[OrderedStatistic(items_base=frozenset({'whole wheat pasta'}), items_add=frozenset({'olive oil'}), confidence=0.2714932126696833, lift=4.122410097642296)]),
# RelationRecord(items=frozenset({'shrimp', 'pasta'}), support=0.005065991201173177, ordered_statistics=[OrderedStatistic(items_base=frozenset({'pasta'}), items_add=frozenset({'shrimp'}), confidence=0.3220338983050847, lift=4.506672147735896)])]


# base_item , add_item , support , confidence , lift 


rules[0][0]
# frozenset({'chicken', 'light cream'})

# Support 
rules[0][1]

# base_item 
rules[0][2][0][0]

# add_item
rules[0][2][0][1]

# Confidence value
rules[0][2][0][2]

# Lift Values
rules[0][2][0][3]



base_item = [] 
add_item = []
support = [] 
confidence = [] 
lift = [] 

for i in range(len(rules)):
    base_item.append(rules[i][2][0][0])
    add_item.append(rules[i][2][0][1])
    confidence.append(rules[i][2][0][2])
    lift.append(rules[i][2][0][3])
    support.append(rules[i][1])
    
d1 = pd.DataFrame(base_item, columns=['Base item'])
d2 = pd.DataFrame(add_item, columns=['Add item'])
d3 = pd.DataFrame(support, columns=['Support'])
d4 = pd.DataFrame(confidence, columns=['Confidence'])
d5 = pd.DataFrame(lift, columns=['Lift'])

df = pd.concat([d1,d2,d3,d4,d5],axis=1)
df

#               Base item     Add item   Support  Confidence      Lift
# 0           light cream      chicken  0.004533    0.290598  4.843951
# 1  mushroom cream sauce     escalope  0.005733    0.300699  3.790833
# 2                 pasta     escalope  0.005866    0.372881  4.700812
# 3         fromage blanc        honey  0.003333    0.245098  5.164271
# 4         herb & pepper  ground beef  0.015998    0.323450  3.291994
# 5          tomato sauce  ground beef  0.005333    0.377358  3.840659
# 6           light cream    olive oil  0.003200    0.205128  3.114710
# 7     whole wheat pasta    olive oil  0.007999    0.271493  4.122410
# 8                 pasta       shrimp  0.005066    0.322034  4.506672

# Here highest support is highest for herb & pepper , ground beef

# Confidence highest is for pasta and escalope ---> the person who buys
# pasta has the highest chance of buying escalope. 

# Lift highest is for fromage blanc and honey.


# Based on these support, confidence and lift values we can recommend the add items. 

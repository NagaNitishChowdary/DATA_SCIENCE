# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 07:20:29 2024

@author: Naga Nitish
"""

# OPEN AND READ THE CONTENTS OF THE FILE 
with open('2_apple.txt','r',encoding='utf-8') as file:
    text_data = file.read() 
    
print(text_data)


# ===========================================================================

# Tokenize the text 
from nltk.tokenize import word_tokenize

t1 = word_tokenize(text_data)
print(len(t1))
# 4793

# ============================================================================

# Convert all words to LowerCase

t2 = []
for i in t1:
    t2.append(i.lower())

print(t2)
    
# ===========================================================================

# REMOVE THE STOP-WORDS LIST 

from nltk.corpus import stopwords
stopwords_list = stopwords.words('english')
stopwords_list
# ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
# "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
# 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
# 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 
# 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 
# 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
# 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
# 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
# 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
#  'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
#  'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 
# 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
# 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
# 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
# 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
# "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
# "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
# 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
# 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

t3 = [] 
for i in t2:
    if i not in stopwords_list:
        t3.append(i)
        
print("After removing stopwords: ",len(t3))
# 2972

# ===========================================================================

# REMOVING PUNCTUATION MARKS AS WELL 

import string 
string.punctuation
# '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

t4 = [] 
for i in t3:
    if i not in string.punctuation:
        t4.append(i)
        
print(t4)

print(len(t4))
# 2485

# ==========================================================================

# After removing punctuations also we have some extra symbols as '', ..(2 dots) etc

punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~.....''``'s'''

t5 = []

for i in t4:
    if i not in punctuations:
        t5.append(i)

print(t5)
print('words with out punctuation:',len(t5))
# 2346

# ========================================================================

# Lemmatization 
# refers to the use of vocabulary and morphological analysis of words, aiming 
# to return the base or dictionary form of a word, which is known as lemma. 

# eg. argue, argued, argues, arguing ---> argue

from nltk.stem import WordNetLemmatizer
Lemmatizer = WordNetLemmatizer()

t6 = []
for i in t5:
    t6.append(Lemmatizer.lemmatize(i))
    
print(len(t6))
2346

# ========================================================================

# Count the each from the above words with out punctuation and show up the
# frequencies of every word 

from collections import Counter 

# count the frequency of each string 
word_counts = Counter(t6)

print(word_counts)

# FROM THIS WE CAN UNDERSTAND WHEN USER IS GIVING FEEDBACK WE CAN UNDERSTAND 
# ON WHICH HE IS FOCUSING MORE.

# =======================================================================

# VISUALIZATION ---> words into a graph based on its frequency 
# It will highlight those words where users are highly used 

# pip install wordcloud

from wordcloud import WordCloud 
import matplotlib.pyplot as plt 

# Create a dictionary to store the word frequency 
word_counts = {} 
for i in t6:
    if i not in word_counts:
        word_counts[i] = 1 
    else:
        word_counts[i] += 1 

print(len(word_counts))
# 1066

# Create the Word Cloud 
wordcloud = WordCloud(width=800,height=600,background_color='white').generate_from_frequencies(word_counts)

# Display the Word Cloud 
plt.figure(figsize=(10,7))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()

# =======================================================================

# From the word cloud remove some words which are not giving any meaning for us 

t7 = [word for word in t6 if word not in ['apple','appl','year','product','air','laptop','macbook','mac','work','use','pro']]

# Same above code 
word_counts = {} 
for i in t7:
    if i not in word_counts:
        word_counts[i] = 1 
    else:
        word_counts[i] += 1 

print(len(word_counts))
# 1057

# Create the Word Cloud 
wordcloud = WordCloud(width=800,height=600,background_color='white').generate_from_frequencies(word_counts)

# Display the Word Cloud 
plt.figure(figsize=(10,7))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


# ===========================================================================


# Here User used "good" ---> the user may used "not good", "very good". 


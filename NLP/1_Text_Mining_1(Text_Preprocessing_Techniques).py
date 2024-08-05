# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 20:01:09 2024

@author: Naga Nitish
"""

import nltk 
nltk.download("all")


# FEEDBACK GIVEN ON THE LAPTOP
text = "I got a great deal on this laptop, which I have been using for almost a year now. I got it not only on discount but also was able to exchange my old laptop with this! Laptop works really well, it was a good purchase!"


# TOKENIZATION ---> is breaking the documents/sentences into chanks called Tokens.
from nltk.tokenize import word_tokenize

# Tokenize the text 
t1 = word_tokenize(text)
t1
# ['I','got','a','great','deal','on','this','laptop',',','which','I','have','been','using','for','almost','a','year','now','.','I','got','it','not','only','on','discount','but','also','was','able','to','exchange','my','old','laptop','with','this','!','Laptop','works','really','well',',','it','was','a','good','purchase','!']

len(t1)
# 50 

# ===============================================================

# Convert all words to lowercase ---> Normalization of text 

t2 = [word.lower() for word in t1]
t2

# =================================================================


# STOP-WORDS LIST 

from nltk.corpus import stopwords
stopwords_list = stopwords.words('english')
print(len(stopwords_list))
# 179

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

# =========================================================================

# REMOVING STOP-WORDS FROM T2 

t3 = []
for i in t2:
    if i not in stopwords_list:
        t3.append(i)
        
print(t3)
# ['got', 'great', 'deal', 'laptop', ',', 'using', 'almost', 'year', '.', 'got',
# 'discount', 'also', 'able', 'exchange', 'old', 'laptop', '!', 'laptop', 'works',
# 'really', 'well', ',', 'good', 'purchase', '!']

len(t3)
# 25 ---> 50% words are removed 

# ============================================================================

# REMOVING PUNCTUATION MARKS AS WELL 

import string 
string.punctuation
# '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

t4 = [] 
for i in t3:
    if i not in string.punctuation:
        t4.append(i)
        
print(t4)
# ['got', 'great', 'deal', 'laptop', 'using', 'almost', 'year', 'got', 'discount', 
#  'also', 'able', 'exchange', 'old', 'laptop', 'laptop', 'works', 'really', 'well',
# 'good', 'purchase']

print(len(t4))
# 20 


# =============================================================================

# Count the each from the above words with out punctuation and show up the
# frequencies of every word 

from collections import Counter 

# count the frequency of each string 
word_counts = Counter(t4)

print(word_counts)
# Counter({'laptop': 3, 'got': 2, 'great': 1, 'deal': 1, 'using': 1, 'almost': 1, 
# 'year': 1, 'discount': 1, 'also': 1, 'able': 1, 'exchange': 1, 'old': 1,
#  'works': 1, 'really': 1, 'well': 1, 'good': 1, 'purchase': 1})


# FROM THIS WE CAN UNDERSTAND WHEN USER IS GIVING FEEDBACK WE CAN UNDERSTAND 
# ON WHICH HE IS FOCUSING MORE.  

# ===========================================================================


# STEMMING 
# is a crude heuristic procsess that chops off the ends of words with out
# considering linguistic features of words.

# eg. argue, argued, argues, arguing ---> argu 


words = ['run', 'runner', 'running', 'ran', 'runs']

from nltk.stem.porter import PorterStemmer
p_stemmer = PorterStemmer() 

for i in words:
    print(i + '--->' + p_stemmer.stem(i))
    
# run--->run
# runner--->runner
# running--->run
# ran--->ran
# runs--->run

# ==========================================================================


# lEMMATIZATION 
# refers to the use of vocabulary and morphological analysis of words, aiming 
# to return the base or dictionary form of a word, which is known as lemma. 

# eg. argue, argued, argues, arguing ---> argue

from nltk.stem import WordNetLemmatizer
Lemmatizer = WordNetLemmatizer()

for i in words:
    print(f"{i:{10}} ---> {Lemmatizer.lemmatize(i)}")
    
# run        ---> run
# runner     ---> runner
# running    ---> running
# ran        ---> ran
# runs       ---> run

# ==========================================================================

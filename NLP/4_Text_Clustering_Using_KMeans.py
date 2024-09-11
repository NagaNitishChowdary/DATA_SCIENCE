# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 00:06:10 2024

@author: Naga Nitish
"""

import numpy as np
import pandas as pd


from sklearn.datasets import fetch_20newsgroups

categories = [
 'comp.os.ms-windows.misc',
 'rec.sport.hockey',
 'soc.religion.christian',
]


# LOADING THE DATA 

dataset = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, 
                             remove=('headers', 'footers', 'quotes'))
data = {'text': dataset.data, 'target': dataset.target}
df = pd.DataFrame(data)

df

"""
                                                   text  target
0     \n   >>So good that there isn't any diff wheth...       0
1     \n\nWell I don't see any smileys here.  I am t...       1
2     \n\nI haven't heard any news about ASN carryin...       1
3     well, the subject says just about all I intend...       0
4     \n   Just a quick question. If Mary was Immacu...       2
                                                ...     ...
1785  I find it interesting that cls never answered ...       2
1786  \nDon't you Americans study history...the Fren...       1
1787  \n\tJesus was born a Jew.  We have biblical ac...       2
1788  09 Apr 93, Jill Anne Daley writes to All:\n\n ...       2
1789  Hi all,\n\nhas anybody tried to compile CTRLTE...       0
"""

# =============================================================================

# PREPROCESSING THE DATA 

# =============================================================================

# REMOVING LINKS 

import re

for i in range(df.shape[0]):
  df['text'][i] = re.sub(r'http\S+', '', df['text'][i])
  
df

"""
                                                   text  target
0     \n   >>So good that there isn't any diff wheth...       0
1     \n\nWell I don't see any smileys here.  I am t...       1
2     \n\nI haven't heard any news about ASN carryin...       1
3     well, the subject says just about all I intend...       0
4     \n   Just a quick question. If Mary was Immacu...       2
                                                ...     ...
1785  I find it interesting that cls never answered ...       2
1786  \nDon't you Americans study history...the Fren...       1
1787  \n\tJesus was born a Jew.  We have biblical ac...       2
1788  09 Apr 93, Jill Anne Daley writes to All:\n\n ...       2
1789  Hi all,\n\nhas anybody tried to compile CTRLTE...       0

[1790 rows x 2 columns]
"""

# =============================================================================

# REMOVE SPECIAL CHARACTERS AND NUMBERS

for i in range(df.shape[0]):
  df['text'][i] = re.sub("[^A-Za-z]+", " ", df['text'][i])

df  

"""
                                                   text  target
0      So good that there isn t any diff whether or ...       0
1      Well I don t see any smileys here I am trying...       1
2      I haven t heard any news about ASN carrying a...       1
3     well the subject says just about all I intende...       0
4      Just a quick question If Mary was Immaculatel...       2
                                                ...     ...
1785  I find it interesting that cls never answered ...       2
1786   Don t you Americans study history the French ...       1
1787   Jesus was born a Jew We have biblical account...       2
1788   Apr Jill Anne Daley writes to All JAD What ex...       2
1789  Hi all has anybody tried to compile CTRLTEST f...       0

[1790 rows x 2 columns]
"""
  
# =============================================================================

# REMOVING STOP WORDS

from nltk.corpus import stopwords

stopwords_list = stopwords.words('english')
stopwords_list
"""
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
"""

from nltk.tokenize import word_tokenize

for i in range(df.shape[0]):
    token = word_tokenize(df['text'][i])
    kt = ""
    for word in token:
        if word not in stopwords_list:
            kt += word 
            kt += " "
    df['text'][i] = kt[:-1]
    
df    
    
"""
                                                   text  target
0     So good diff whether ATManager turned Is worth...       0
1     Well I see smileys I trying figure poster dog ...       1
2     I heard news ASN carrying games local cable st...       1
3     well subject says I intended ask Is way insert...       0
4     Just quick question If Mary Immaculately conci...       2
                                                ...     ...
1785  I find interesting cls never answered question...       2
1786  Don Americans study history French settled Nor...       1
1787  Jesus born Jew We biblical accounts mother anc...       2
1788  Apr Jill Anne Daley writes All JAD What exactl...       2
1789  Hi anybody tried compile CTRLTEST MFC SAMPLES ...       0

[1790 rows x 2 columns]
"""

# =============================================================================

# LEMMATIZATION

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

for i in range(df.shape[0]):
  tokenized_list = word_tokenize(df['text'][i])
  lemmatized_doc = ""
  for word in tokenized_list:
    lemmatized_doc = lemmatized_doc + " " + lemmatizer.lemmatize(word)
  df['text'][i] = lemmatized_doc

df

"""
                                                   text  target
0      So good diff whether ATManager turned Is wort...       0
1      Well I see smiley I trying figure poster dog ...       1
2      I heard news ASN carrying game local cable st...       1
3      well subject say I intended ask Is way insert...       0
4      Just quick question If Mary Immaculately conc...       2
                                                ...     ...
1785   I find interesting cl never answered question...       2
1786   Don Americans study history French settled No...       1
1787   Jesus born Jew We biblical account mother anc...       2
1788   Apr Jill Anne Daley writes All JAD What exact...       2
1789   Hi anybody tried compile CTRLTEST MFC SAMPLES...       0

[1790 rows x 2 columns]
"""


# =============================================================================

# FITTING THE MODEL 

from sklearn.feature_extraction.text import TfidfVectorizer

# sublinear_tf=True parameter controls how term frequencies are calculated.

vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=3)
X = vectorizer.fit_transform(df['text']).toarray()


# =============================================================================


# CLUSTERING USING K-MEANS 

from sklearn.cluster import KMeans

km = KMeans(n_clusters=len(categories), init='k-means++', max_iter=100)
km.fit(X)

# =============================================================================

# METRICS CALCULATION

from sklearn import metrics

from sklearn.metrics import adjusted_rand_score 

# ADJUSTED RAND INDEX(ARI) ---> [-1 to 1] ---> measures the similarity between 
# the predicted clusters and the ground truth labels
print("Adjusted Rand Score ", adjusted_rand_score(df['target'], km.labels_))

# Adjusted Rand Score  0.5847887350485512


from sklearn.metrics import normalized_mutual_info_score
# NORMALIZED MUTUAL INFORMATION(NMI) ---> [0 to 1] ---> measures the mutual 
# information between the predicted clusters and the ground truth labels
print(normalized_mutual_info_score(df['target'], km.labels_))

# Normalized Mutual Info Score  0.6249991378298093


from sklearn.metrics import fowlkes_mallows_score
# FOWLKES-MALLOWS INDEX(FMI) ---> [0 to 1] ---> measures the geometric mean of 
# the precision and recall of the predicted clusters with respect to the labels
print("Fowlkes Mallows Score ",fowlkes_mallows_score(df['target'], km.labels_))
# Fowlkes Mallows Score  0.730122655745994


# =============================================================================

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:48:46 2024

@author: Naga Nitish
"""

import numpy as np 
import pandas as pd 

df = pd.read_csv('5_spam.csv', encoding='latin-1')
df.head()

"""
     v1  ... Unnamed: 4
0   ham  ...        NaN
1   ham  ...        NaN
2  spam  ...        NaN
3   ham  ...        NaN
4   ham  ...        NaN

[5 rows x 5 columns]
"""

# =============================================================================

df.columns
"""
Index(['v1', 'v2', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], dtype='object')
"""

# REMOVING UNNECESSARY COLUMNS

df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.head()

"""
     v1                                                 v2
0   ham  Go until jurong point, crazy.. Available only ...
1   ham                      Ok lar... Joking wif u oni...
2  spam  Free entry in 2 a wkly comp to win FA Cup fina...
3   ham  U dun say so early hor... U c already then say...
4   ham  Nah I don't think he goes to usf, he lives aro...
"""

# =============================================================================

df.columns = ['labels','text']
df.head()

"""
  labels                                               text
0    ham  Go until jurong point, crazy.. Available only ...
1    ham                      Ok lar... Joking wif u oni...
2   spam  Free entry in 2 a wkly comp to win FA Cup fina...
3    ham  U dun say so early hor... U c already then say...
4    ham  Nah I don't think he goes to usf, he lives aro...
"""

# =============================================================================

# CHECK IF THERE ANY NULL VALUES IN THE BOTH COLUMNS 

df['labels'].isna().sum()
# 0 

df['text'].isna().sum()
# 0 

# =============================================================================

# Number of Spam mails and Ham mails in the dataset 

df['labels'].value_counts()

"""
ham     4825
spam     747
Name: count, dtype: int64
"""

# =============================================================================

# REMOVE LINKS FROM THE TEXT && REMOVE SPECIAL CHARACTERS AND NUMBERS

import re 

for i in range(df.shape[0]):
    df['text'][i] = re.sub(r'http\S+','',df['text'][i])
    df['text'][i] = re.sub('[^A-Za-z ]','',df['text'][i])
    
    
df.head()

"""
  labels                                               text
0    ham  Go until jurong point  crazy   Available only ...
1    ham                      Ok lar    Joking wif u oni   
2   spam  Free entry in   a wkly comp to win FA Cup fina...
3    ham  U dun say so early hor    U c already then say   
4    ham  Nah I don t think he goes to usf  he lives aro...
"""

# =============================================================================

# REMOVE STOP WORDS FROM THE TEXT  && NORMALIZING THE TEXT

import nltk

from nltk.corpus import stopwords

stopwords_list = stopwords.words('english')
stopwords_list

"""
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
"you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 
'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 
'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 
'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
"couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
"hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
"""

from nltk.tokenize import word_tokenize

for i in range(df.shape[0]):
    token = word_tokenize(df['text'][i])
    sentence = ""
    for word in token:
        if word not in stopwords_list:
            word = word.lower()
            sentence += word
            sentence += " "
    
    df['text'][i] = sentence[:-1]
    

df.head()

"""
  labels                                               text
0    ham  go jurong point crazy available bugis n great ...
1    ham                            ok lar joking wif u oni
2   spam  free entry wkly comp win fa cup final tkts st ...
3    ham                u dun say early hor u c already say
4    ham      nah i dont think goes usf lives around though
"""

# =============================================================================

# LEMMATIZATION

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

for i in range(1) : #df.shape[0]):
    tokenized_list = word_tokenize(df['text'][i])
    lemmatized_doc = "" 
    for word in tokenized_list:
        lemmatized_doc += lemmatizer.lemmatize(word)
        lemmatized_doc += " "
        
    df['text'][i] = lemmatized_doc[:-1]
    
df.head()
    
"""
  labels                                               text
0    ham  go jurong point crazy available bugis n great ...
1    ham                            ok lar joking wif u oni
2   spam  free entry wkly comp win fa cup final tkts st ...
3    ham                u dun say early hor u c already say
4    ham         nah i dont think go usf life around though
"""

# =============================================================================

# SPLITING THE DATA FOR TRAINING AND TESTING 

from sklearn.model_selection import train_test_split 

X = df['text']
Y = df['labels']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30, random_state=42) 

X_train.shape
# (3900,)

X_test.shape
# (1672,)

# =============================================================================


# FEATURE EXTRACTION 

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

print("X_train_cv.shape: ", X_train_cv.shape)
# X_train_cv.shape:  (3900, 6939)

print("X_test_cv.shape: ", X_test_cv.shape)
# X_test_cv.shape:  (1672, 6939)

# =============================================================================

# MODEL FITTING 

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()

LR.fit(X_train_cv, Y_train)

Y_Pred_train = LR.predict(X_train_cv)
Y_Pred_test = LR.predict(X_test_cv)


# =============================================================================

# METRICS 

from sklearn.metrics import accuracy_score, r2_score

print("Training Accuracy: ", accuracy_score(Y_train, Y_Pred_train))
# 99.58%

print("Testing Accuracy: ", accuracy_score(Y_test, Y_Pred_test))
# 97.66%

# =============================================================================

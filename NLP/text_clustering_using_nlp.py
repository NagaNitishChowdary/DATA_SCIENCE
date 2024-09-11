# -*- coding: utf-8 -*-
"""Text_Clustering_Using_NLP.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1g4AaflN0WklPX6OIEGNVd4X_TqVkGKdm
"""

from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import word_tokenize # Used to extract words from documents
from nltk.stem import WordNetLemmatizer # Used to lemmatize words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans

import sys
from time import time

import pandas as pd
import numpy as np

# Selected 3 categories from the 20 newsgroups dataset

categories = [
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

print("Loading 20 newsgroups dataset for categories:")
print(categories)

# fetch_20newsgroups() fetches the "20 Newsgroups" dataset

df = fetch_20newsgroups()
df

df['data']

print(len(df['data']))

type(df['data'])

type(df)

len(df['data'])

df.keys()

df['filenames']

df['target']

len(df['target_names'])

# subset ---> The subset parameter specifies which portion of the dataset you want to load. It can take values like 'train', 'test', or 'all'
# default value of subset parameter is 'train' (only 1554 out of 2588 loaded)
# here we are taking setting subset value to 'all' (to load both training and testing data)

# categories ---> Defines the categories of the newsgroups to be loaded.

# shuffle ---> This parameter specifies whether to shuffle the order of the documents

# remove ---> Tells the function to remove certain parts of the text that are usually not relevant to the content itself like
# headers(from, subject)

df1 = fetch_20newsgroups(subset='all', categories=categories, shuffle=False, remove=('headers', 'footers', 'quotes'))
df1

np.unique(df1['target'])  # we have selected only 3 categories, so there are only 3 targets

labels = df1.target
print(labels)

"""**PREPROCESSING THE DATA**"""

# REMOVE LINKS FROM DATA

# re.sub() is used to search for a pattern in the text and replace it with a specified string.

import re

for i in range(len(df1['data'])):
  df1['data'][i] = re.sub(r'http\S+', '', df1['data'][i])

# REMOVE SPECIAL CHARACTERS AND NUMBERS

for i in range(len(df1['data'])):
  df1['data'][i] = re.sub("[^A-Za-z]+", " ", df1['data'][i])

import nltk
nltk.download('punkt')
nltk.download('wordnet')

"""PERFORMING **LEMMATIZATION**"""

# The process of converting a word to its dictionary (base) form, or lemma, by removing inflectional endings and returning the base or
# dictionary form of a word.

lemmatizer = WordNetLemmatizer()

for i in range(len(df1['data'])):
  tokenized_list = word_tokenize(df1['data'][i])
  lemmatized_doc = ""
  for word in tokenized_list:
    lemmatized_doc = lemmatized_doc + " " + lemmatizer.lemmatize(word)
  df1['data'][i] = lemmatized_doc

print(df1.data[0])

"""**MODEL FITTING** USING **TF-IDF**"""

# Converting text data into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique.

# strip_accents='unicode' ---> any accented characters will be converted to their ASCII equivalents using Unicode normalization.
# ("café" would become "cafe")

# stop_words = 'english' ---> removes common English stop words(like "the", "and", "in") from the text before vectorization

# min_df ---> sets the minimum document frequency.
# min_df=2 means that any word that appears in fewer than 2 documents in the corpus(collection of documents) will be ignored(not included
# in the vocabulary of the vectorizer)
# main use is to remove the spelling mistakes

vectorizer = TfidfVectorizer(strip_accents='unicode', stop_words='english', min_df=2)

# fit_transform ---> fits the TfidfVectorizer to the text data df1.data, which means it learns the vocabulary (i.e., the set of all unique words)
# from the input documents and computes the IDF (Inverse Document Frequency) values for all the terms in the corpus.

# It then transforms the text data into a sparse matrix of TF-IDF features. Each row of this matrix represents a document, and each column
# represents a term (word) from the vocabulary.

# The value at a given position in this matrix represents the TF-IDF score of a word in a particular document, which reflects how important
# that word is in the document compared to its frequency across the entire corpus(collection of documents).

X = vectorizer.fit_transform(df1.data)

X.shape

# Clustering using standard k-means (groups text data into clusters based on their similarity.)

# max_iter ---> specifies the maximum number of iterations the algorithm will run for each single run. If the algorithm hasn't converged
# (i.e., the clusters haven't stabilized) after 100 iterations, it will stop ---> prevents from infinite loop)

# init ---> The init parameter determines how the initial cluster centers are selected. k-means++ is a method that chooses initial cluster
# centers in a way that speeds up convergence and often leads to better clustering results.

km = KMeans(n_clusters=len(np.unique(df1['target'])), init='k-means++', max_iter=100)
km.fit(X)

"""**METRICS**"""

print(km.labels_)
print(labels)

cnt = 0
correct = 0
for i in range(len(labels)):
  if km.labels_[i] == labels[i]:
    correct += 1
  cnt += 1

print(correct,cnt)

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score

# ADJUSTED RAND INDEX(ARI) ---> [-1 to 1] ---> measures the similarity between the predicted clusters and the ground truth labels
print("Adjusted Rand Score ", adjusted_rand_score(labels, km.labels_))

# NORMALIZED MUTUAL INFORMATION(NMI) ---> [0 to 1] ---> measures the mutual information between the predicted clusters and the ground truth labels
print("Normalized Mutual Info Score ", normalized_mutual_info_score(labels, km.labels_))

# FOWLKES-MALLOWS INDEX(FMI) ---> [0 to 1] ---> measures the geometric mean of the precision and recall of the predicted clusters with respect to the ground truth labels
print("Fowlkes Mallows Score ",fowlkes_mallows_score(labels, km.labels_))

# IDENTIFYING THE 10 MOST RELAVENT TERMS IN EACH CLUSTER

centroids = km.cluster_centers_.argsort()[:,::-1]  # Indices of largest centroids entries in descending order
terms = vectorizer.get_feature_names_out()

for i in range(len(np.unique(df1['target']))):
  print("Cluster %d: "%i,end=' ')
  for ind in centroids[i,:10]:
    print(' %s' % terms[ind],end=' ')
  print()

"""**VISUALIZATION**"""

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def frequencies_dict(cluster_index):
    if cluster_index > len(np.unique(df1['target'])) - 1:
        return
    term_frequencies = km.cluster_centers_[cluster_index]
    sorted_terms = centroids[cluster_index]
    frequencies = {terms[i]: term_frequencies[i] for i in sorted_terms}
    return frequencies

def makeImage(frequencies):

    wc = WordCloud(background_color="white", max_words=50)
    # generate word cloud
    wc.generate_from_frequencies(frequencies)

    # show
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

for i in range(len(np.unique(df1['target']))):
    freq = frequencies_dict(i)
    makeImage(freq)
    print()
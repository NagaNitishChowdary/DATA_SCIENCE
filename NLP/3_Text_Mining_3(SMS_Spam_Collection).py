"""
Created on Tue Aug  6 20:35:10 2024

@author: Naga Nitish
"""

import pandas as pd
df = pd.read_csv("smsspamcollection.tsv" , sep='\t')
df

# ========================================================================

# label encoding
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df['label'] = le.fit_transform(df['label'])
df.head()

# ========================================================================

# normalization
df['message'] = df['message'].str.lower()
df.head()

# ============================================================================

# removing punctuations  ---> removing quotes, commas etc. 
import string
string.punctuation

def remove_punctuation(text):
  text_nopunct = "".join([char for char in text if char not in string.punctuation])
  return text_nopunct

df['message'] = df['message'].apply(lambda x: remove_punctuation(x))
df.head()

# ==============================================================================

# applying lemmatization
#import nltk
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

df['message'] = df['message'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
df.head()

# =============================================================================

# remvoing stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

df['message'] = df['message'].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))
df.head()

# =============================================================================

#split the variables
Y = df["label"]
x = df['message']

# =============================================================================

# feature extraction
# Tf (Term Frequency) , idf(Inverse document frequency)  ---> eg. good -> 3 (times)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(x)

X

# =============================================================================S

# data partition
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_test shape: {Y_test.shape}")

#==============================================================================

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, Y_train)
#==============================================================================
# prompt: predict the model using training data and test data

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

#==============================================================================
# prompt: calculate the accuracy score for both training and testing

from sklearn.metrics import accuracy_score

accuracy_train = accuracy_score(Y_train, y_pred_train)
accuracy_test = accuracy_score(Y_test, y_pred_test)

print(f"Accuracy on training data: {accuracy_train.round(2)}")
print(f"Accuracy on testing data: {accuracy_test.round(2)}")


#==============================================================================

from sklearn.neural_network import MLPClassifier
model = MLPClassifier()
model.fit(X_train, Y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(Y_train, y_pred_train)
accuracy_test = accuracy_score(Y_test, y_pred_test)
print(f"Accuracy on training data: {accuracy_train.round(2)}")
print(f"Accuracy on testing data: {accuracy_test.round(2)}")

# =============================================================================

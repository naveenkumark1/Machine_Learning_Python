# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 18:40:21 2019

@author: BestintownACER1
"""

#Import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
import numpy as np

#assigning predictor and target variables
x= np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])


#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(x, y)

#Predict Output 
predicted= model.predict([[1,2],[3,4]])
print (predicted)

Output: ([3,4])


# -------------------------- Simple program for Naive Bayes -----------------------------------

from sklearn.datasets import make_blobs
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)

fig, ax = plt.subplots()

ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
ax.set_title('Naive Bayes Model', size=14)

xlim = (-8, 8)
ylim = (-15, 5)

xg = np.linspace(xlim[0], xlim[1], 60)
yg = np.linspace(ylim[0], ylim[1], 40)
xx, yy = np.meshgrid(xg, yg)
Xgrid = np.vstack([xx.ravel(), yy.ravel()]).T

for label, color in enumerate(['red', 'blue']):
    mask = (y == label)
    mu, std = X[mask].mean(0), X[mask].std(0)
    P = np.exp(-0.5 * (Xgrid - mu) ** 2 / std ** 2).prod(1)
    Pm = np.ma.masked_array(P, P < 0.03)
    ax.pcolorfast(xg, yg, Pm.reshape(xx.shape), alpha=0.5,
                  cmap=color.title() + 's')
    ax.contour(xx, yy, P.reshape(xx.shape),
               levels=[0.01, 0.1, 0.5, 0.9],
               colors=color, alpha=0.2)
    
ax.set(xlim=xlim, ylim=ylim)

#fig.savefig('figures/05.05-gaussian-NB.png')


##############################
import os
os.chdir("C:/Users/BestintownACER1/Desktop/ML_P/Naive_Bayes_Text_Classifier_SPAM_NOSPAM")

# Spam Classification 

import pandas as pd

# Read the Data, Tab Seperated & create Labels

df = pd.read_table('SMSSpamCollection',  
                   sep='\t', 
                   header=None,
                   names=['label', 'message'])

#####################   PRE PROCESSING 

# Map labels to Numbers - Convert String to Binary values 

df['label'] = df.label.map({'ham': 0, 'spam': 1}) 

# Convert all characters in to lower cases 

df['message'] = df.message.map(lambda x: x.lower())

# Remove the punctuation marks 

df['message'] = df.message.str.replace('[^\w\s]', '')  

# Download NLTK for the process of Tokenization, Stemming, Lemmatization, Creating a Documentation Term Matrix 

import nltk  
#nltk.download()  

################# Tokenization : Convert the sentences in to Words

df['message'] = df['message'].apply(nltk.word_tokenize)  

################# Performing Stemming using : Porter Stemmer method

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

df['message'] = df['message'].apply(lambda x: [stemmer.stem(y) for y in x])  

##############

from sklearn.feature_extraction.text import CountVectorizer

# This converts the list of words into space-separated strings

df['message'] = df['message'].apply(lambda x: ' '.join(x))

count_vect = CountVectorizer()  
counts = count_vect.fit_transform(df['message'])  

##################

# Create a TF IDF to know the weighnts of every word int he DTM

from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer().fit(counts)

counts = transformer.transform(counts) 

### Model Building Phase #### 
##############################

# Split the data in to Trining and the Testing Data Set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(counts, df['label'], test_size=0.1, random_state=69) 

######################

#  Building the model using the Naive Bayes classifier on the Train Data 

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(alpha=0.5).fit(X_train, y_train)  

# Alpha <1 - laplace smoothing , >1 Lidstone Smoothing

#################

# Evaluating the Model on the Test Data Set 

import numpy as np

predicted = model.predict(X_test)

print(np.mean(predicted == y_test))  

#########  Confusion matrix to chek the perfomance of the Model

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, predicted))  

###############


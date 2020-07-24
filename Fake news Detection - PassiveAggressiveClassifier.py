#!/usr/bin/env python
# coding: utf-8

# # Fake News Detection - PassiveAggressiveClassifier

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import dataset
data = pd.read_csv( r"C:\Users\Dell\Desktop\Machine Learning\Sample Dataset\news.csv")

#Check the size of datset
data.shape

#Displaying first 5 dataset entries
data.head(5)

# Get the titles
title = data.label
title.head()

#Check if any null value is present
data.isnull().values.any()


#Splitting Training and Testing datset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data['text'], title, test_size = 0.30, random_state = 10)


#Initialize a TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(stop_words='english', max_df = 0.7)

# Fit and transform train set, transform test set
tfidf_train = tfidf.fit_transform(X_train) 
tfidf_test = tfidf.transform(X_test)


#Initializing a PassiveAggressiveClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
p = PassiveAggressiveClassifier(max_iter=50)
p.fit(tfidf_train, Y_train)

#Predicting Accuracy of model
predict = p.predict(tfidf_test)
from sklearn import metrics
print("Accuracy  = {0:.3f}".format(metrics.accuracy_score(Y_test, predict)))


#Building a confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, predict,labels=['FAKE','REAL'])

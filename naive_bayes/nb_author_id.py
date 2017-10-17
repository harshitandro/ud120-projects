#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB as nb


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
classifier = nb()
t_start = time()
classifier.fit(features_train,labels_train)
print("Classifier training  took :{} sec".format(round(time()-t_start),4))
t_start = time()
score = classifier.score(features_test,labels_test)
t_end = time()
print("Accuracy : {}".format(score))
print("Prediction took : {} sec".format(round(t_end-t_start,4)))
#########################################################



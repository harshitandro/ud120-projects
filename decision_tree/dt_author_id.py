#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from time import time
from sklearn import tree
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

classifier = tree.DecisionTreeClassifier(min_samples_split=40)
print("No. of features :{}".format(len(features_test[0])))
#########################################################
### your code goes here ###

t_start = time()
classifier.fit(features_train,labels_train)
print("Classifier training took :{} sec".format(round(time()-t_start),4))
t_start = time()
pred = classifier.predict(features_test)
t_end = time()
#counter = 0
#for x in pred:
#    if x == 1 :
#        counter +=1
#print("Chris's emails identified :%d"%counter)
score = classifier.score(features_test,labels_test)
print("Accuracy : {}".format(score))
print("Prediction took : {} sec".format(round(t_end-t_start,4)))

#########################################################



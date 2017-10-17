#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from time import time
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

classifier = SVC(kernel="rbf",cache_size=2048,C=10000)

#########################################################
### your code goes here ###
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

t_start = time()
classifier.fit(features_train,labels_train)
print("Classifier training took :{} sec".format(round(time()-t_start),4))
t_start = time()
pred = classifier.predict(features_test)
counter = 0
for x in pred:
    if x == 1 :
        counter +=1
print("Chris's emails identified :%d"%counter)
score = classifier.score(features_test,labels_test)
t_end = time()
print("Accuracy : {}".format(score))
print("Prediction took : {} sec".format(round(t_end-t_start,4)))

#########################################################



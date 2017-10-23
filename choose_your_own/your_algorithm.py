#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from sklearn import naive_bayes as nb
from sklearn import svm
from sklearn import tree

from time import time
print("No. of features : {}".format(len(features_test[0])))
print("No. of training samples : {}".format(len(features_train)/len(features_train[0])))
print("No. of testing samples : {}".format(len(features_test)/len(features_test[0])))

def NBTest():
    #Naive Bayes
    print("\nNaive Bayes Data :")
    clf = nb.GaussianNB()
    start_time = time()
    clf.fit(features_train,labels_train)
    end_time = time()
    print("Training took  : {} sec".format(round(end_time-start_time,3)))
    print("Accuracy : {}".format(clf.score(features_test,labels_test)))
    try:
        prettyPicture(clf, features_test, labels_test)
    except NameError:
        pass

def SVCTest():
    # SVC with rbf kernel
    print("\nSVC Data :")
    clf = svm.SVC(kernel="rbf",cache_size=2048,C=100,gamma=10)
    start_time = time()
    clf.fit(features_train,labels_train)
    end_time = time()
    print("Training took  : {} sec".format(round(end_time-start_time,3)))
    print("Accuracy : {}".format(clf.score(features_test,labels_test)))
    try:
        prettyPicture(clf, features_test, labels_test)
    except NameError:
        pass


def DecisionTreeTest():
    # Decision Tree
    print("\nDecision Tree Data :")
    clf = tree.DecisionTreeClassifier(min_samples_split=40,criterion="entropy")
    start_time = time()
    clf.fit(features_train,labels_train)
    end_time = time()
    print("Training took  : {} sec".format(round(end_time-start_time,3)))
    print("Accuracy : {}".format(clf.score(features_test,labels_test)))
    try:
        prettyPicture(clf, features_test, labels_test)
    except NameError:
        pass
NBTest()
SVCTest()
DecisionTreeTest()
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 00:46:55 2019

@author: Ray
"""
from __future__ import print_function
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from proprocess_data import get_ckd_dataset
from sklearn.metrics import f1_score


X_train, X_test, y_train, y_test = get_ckd_dataset(normalized=True,test_ratio=0.2)

linear_svc = SVC(kernel='linear')
rbf_svc = SVC(kernel='rbf')
rf = RandomForestClassifier()


linear_svc.fit(X_train,y_train)
rbf_svc.fit(X_train,y_train)
rf.fit(X_train,y_train)

print("F measure on training set:")

print("Linear SVC = {}".format(f1_score(y_train,linear_svc.predict(X_train))))
print("RBF SVC = {}".format(f1_score(y_train,rbf_svc.predict(X_train))))
print("RandomForest = {}".format(f1_score(y_train,rf.predict(X_train))))

print("-"*30)
print("F measure on test set:")
print("Linear SVC = {}".format(f1_score(y_test,linear_svc.predict(X_test))))
print("RBF SVC = {}".format(f1_score(y_test,rbf_svc.predict(X_test))))
print("RandomForest = {}".format(f1_score(y_test,rf.predict(X_test))))

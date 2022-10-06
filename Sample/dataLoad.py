#!/usr/bin/env python3
# coding:utf-8

# Brain-Computer Interface Demo
# v.1.0
# John LaRocco
# 21 June 2021
# dataPrep.py

# Usage:
# This script is the main demo for an offline linear brain-computer interface.
# It imports EEG files from an OpenBCI Gangion.
# It requires support functions.

# implemented with Python3 on Anaconda

# import basic modules
import numpy as np
import scipy as sp
import pandas as pd

from supportFunctions import featureExtraction, eegFeatureExtraction, eegFeatureReducer, balancedMatrix

from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


# Set parameters
print('Setting parameters...')
fs = 200  # features per second (Hz)
pcti = 99.95  # amplitude percent
lowcut = 1  # filter lower bound (Hz)
highcut = (np.floor(fs/2))  # filter upper bound (Hz)
featureNumber = 3  # number of features to retain
N = 4  # value for N-fold cross-validation

# filenames of EEG files
nameFile1 = 'eegSession1.txt'
nameFile2 = 'eegSession2.txt'

# load data
print('Loading datasets...')
df1 = pd.read_csv(nameFile1, skiprows=6)
df2 = pd.read_csv(nameFile2, skiprows=6)

# perform feature extraction
print('Extracting features...')
featureMatrixA = eegFeatureExtraction(df1, fs, lowcut, highcut, pcti)
featureMatrixB = eegFeatureExtraction(df2, fs, lowcut, highcut, pcti)

# perform feature selection
print('Selecting features...')
topFeatures = eegFeatureReducer(featureMatrixA, featureMatrixB, featureNumber)

featureMatrixA = np.squeeze(featureMatrixA[:, topFeatures])
featureMatrixB = np.squeeze(featureMatrixB[:, topFeatures])

t0 = np.zeros(np.shape(featureMatrixA)[0])
t1 = np.ones(np.shape(featureMatrixB)[0])

totalLength = np.array([len(t0), len(t1)])

# prepare data for classification
print('Preparing for classification...')
s0 = balancedMatrix(featureMatrixA, totalLength)
s1 = balancedMatrix(featureMatrixB, totalLength)

X = np.vstack([s0, s1])
t0 = 0*np.ones([1, len(s0)])
t1 = 1*np.ones([1, len(s1)])

targets = np.hstack([t0, t1])
y = np.transpose(np.ravel(targets))


# compare classifiers
print('Running classifiers...')
clf = QuadraticDiscriminantAnalysis()
print('QDA/LDA Results: ')
scores = cross_val_score(clf, X, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)

clf = LogisticRegression(random_state=0)
print('Logistic Regression Results: ')
scores = cross_val_score(clf, X, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)

clf = GaussianNB()
print('Naive Bayes Results: ')
scores = cross_val_score(clf, X, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)

clf = SVC(gamma=2, C=1)
print('Linear SVM Results: ')
scores = cross_val_score(clf, X, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)

clf = AdaBoostClassifier(n_estimators=1000, random_state=0)
print('AdaBoost Results: ')
scores = cross_val_score(clf, X, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)

clf = MLPClassifier(alpha=2, max_iter=100)
print('MLP Results: ')
scores = cross_val_score(clf, X, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)

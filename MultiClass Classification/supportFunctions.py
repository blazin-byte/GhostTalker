import numpy as np
import scipy as sp
import pandas as pd
from scipy import signal, arange, fft, fromstring, roll
from scipy.signal import butter, lfilter, ricker
import os
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from scipy.stats import stats


def eegFeatureExtraction(df, fs, lowcut, highcut, pcti):
    chan1 = df.iloc[:, 2]
    chan2 = df.iloc[:, 3]
    chan3 = df.iloc[:, 4]
    chan4 = df.iloc[:, 5]

    # rotating the vectors to array
    c1 = np.real(np.asarray(chan1))
    c2 = np.real(np.asarray(chan2))
    c3 = np.real(np.asarray(chan3))
    c4 = np.real(np.asarray(chan4))

    # Normalizing these arrays
    c1 = c1-np.mean(c1)
    c2 = c2-np.mean(c2)
    c3 = c1-np.mean(c3)
    c4 = c4-np.mean(c4)

    c1 = c1[fs::]

    f1 = featureExtraction(c1, fs, lowcut, highcut, pcti)
    features = np.squeeze(np.shape(f1))

    c2 = c2[fs::]
    c3 = c3[fs::]
    c4 = c4[fs::]

    lengthFile = np.floor(np.squeeze(np.shape(c1))/np.float(4*fs))
    lbnds = np.arange(0, (lengthFile-1))
    ubnds = np.arange(1, (lengthFile))
    capper = np.min([len(lbnds), len(ubnds)])
    lbnds = 4*fs*lbnds[0:capper]
    ubnds = 4*fs*ubnds[0:capper]
    featureMatrix = np.zeros((capper, (4*features)))

    for ix in range(0, capper):
        s1 = featureExtraction(
            c1[int(lbnds[ix]):int(ubnds[ix])], fs, lowcut, highcut, pcti)
        s2 = featureExtraction(
            c2[int(lbnds[ix]):int(ubnds[ix])], fs, lowcut, highcut, pcti)
        s3 = featureExtraction(
            c3[int(lbnds[ix]):int(ubnds[ix])], fs, lowcut, highcut, pcti)
        s4 = featureExtraction(
            c4[int(lbnds[ix]):int(ubnds[ix])], fs, lowcut, highcut, pcti)
        featall = np.concatenate((s1, s2, s3, s4), axis=0)
        featall = np.squeeze(featall[0:(4*features)])
        featureMatrix[int(ix), :] = featall
    return (featureMatrix)


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = (lowcut / nyq)
    high = (highcut / nyq)
    if high >= 1:
        high = .99
    if low <= 0:
        low = .001
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def welchProc(data, fs):
    # wsize=round(fs/10)
    f, P = signal.welch(data, fs)
    return f, P


def peakFinder(f, P):
    peakFLoc = np.where(P == np.amax(P))
    peakFLoc = peakFLoc[0]
    peakF = f[peakFLoc]
    vrms = np.sqrt(P.max())
    return peakF, peakFLoc, vrms


def smooth(x, window_len=11, window='hanning'):
    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y


def featureExtraction(data, fs, lowcut, highcut, pcti):
    widths = np.arange(1, 31)
    data[np.isnan(data)] = 0
    data[np.isinf(data)] = 0
    intensityPcti = np.percentile(data, pcti)
    data = data-np.mean(data)
    data = smooth(data.flatten())
    data = butter_bandpass_filter(data, lowcut, highcut, fs, order=4)
    data = signal.cwt(data, signal.ricker, widths)
    f, P = welchProc(data, fs)
    peakFLoc = np.where(P == np.amax(P))
    peakFLoc = peakFLoc[0]
    peakF = f[peakFLoc]
    vrms = np.sqrt(P.max())
    Psum = np.sum(P, axis=1)
    Psum = Psum.flatten()
    # print(np.shape(Psum))
    # print(np.shape(vrms))
    # print(np.shape(peakF))
    # print(np.shape(peakFLoc))
    # print(np.shape(intensityPcti))
    featureVector = np.hstack(
        (Psum.flatten(), vrms, peakF, peakFLoc, intensityPcti))
    # print(featureVector)
    featureVector[np.isnan(featureVector)] = 0
    featureVector[np.isinf(featureVector)] = 0
    # print(featureVector)
    return featureVector


def balancedMatrix(a, totalLength):
    maxLen = np.max(totalLength)
    minLen = np.min(totalLength)
    ratioL = np.floor(maxLen/minLen)
    finalR = np.ceil(minLen*ratioL)
    aT = np.copy(a)
    features = np.shape(a)[1]
    aTT = np.zeros([1, features])
    for ii in range(0, int(ratioL)):
        aTT1 = np.copy(aT)
        aTT = np.vstack([aTT1, aTT])
        aTT = np.squeeze(aTT)

    aTT = aTT[0:(maxLen-1), :]
    aTT = np.squeeze(aTT)
    return (aTT)


def eegFeatureReducer(featureMatrixA, featureMatrixB, featureNumber):
    m0 = np.mean(featureMatrixA, axis=0)
    m1 = np.mean(featureMatrixB, axis=0)
    distancesVec = np.abs(m0-m1)
    tempR = np.argpartition(-distancesVec, featureNumber)
    resultArgs = tempR[:featureNumber]
    topFeatures = np.flip(resultArgs)

    return (topFeatures)


def featureReducer(Xf, yf, features):
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, features, step=15)
    selector = selector.fit(Xf, yf)
    topFeatures = np.where(selector.ranking_ == 1)
    Xnew = np.squeeze(Xf[:, topFeatures])
    return (Xnew, topFeatures)


def crossValClass(clf, X, y, xfold):
    scores = cross_val_score(clf, X, y, cv=xfold)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(clf, X, y, cv=xfold, scoring='f1_macro')
    print("F1 Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# An element is one combination of trial, phoneme, and subject


def vectorizeElement(df):
    df

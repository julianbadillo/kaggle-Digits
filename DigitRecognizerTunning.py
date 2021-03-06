#!/usr/bin/python
"""
    A class for Recognizing Digits
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
from matplotlib import pyplot as plt
import random
from math import sqrt, ceil
from utils import *
from DigitRecognizer import DigitRecognizer

bits = 14

def tuneL2():
    #load data
    X_tr, y_tr = loadTrainData(f='data/train_14x14.csv')
    m, n = X_tr.shape
    
    acc = []
    l2s = [i for i in range(20,41)]
    for l2 in l2s:
        dig = DigitRecognizer(bits*bits, l2, 10)
        dig.setTrainParams(maxiter=250)
        
        #train
        dig.train(X_tr, y_tr)
        
        #check predictions
        pred = dig.predict(X_tr)
        #accuracy
        acc.append( (pred == y_tr.flatten()).sum()* 100.0 / m )
        print "Accuracy = %s%%"%acc[-1]
        
    plt.plot(l2s, acc)
    plt.xlabel("L2")
    plt.ylabel("accuracy")
    plt.savefig('data/tuneL2.png')
    plt.close()

def tuneEpsilon():
    #load data
    X_tr, y_tr = loadTrainData(f='data/train_14x14.csv')
    m, n = X_tr.shape
    
    acc = []
    ls = np.linspace(0.2, 0.6, 10)
    for eps in ls:
        dig = DigitRecognizer(bits*bits, 27, 10)
        dig.setTrainParams(epsilon=eps, maxiter=100)
        
        #train
        dig.train(X_tr, y_tr)
        
        #check predictions
        pred = dig.predict(X_tr)
        #accuracy
        acc.append( (pred == y_tr.flatten()).sum()* 100.0 / m )
        print "Accuracy = %s%%"%acc[-1]

    plt.plot(ls, acc)
    plt.xlabel("Epsilon")
    plt.ylabel("accuracy")
    plt.savefig('tuneEpsilon.png')
    plt.close()

def tuneLambda():
    #load data
    X_tr, y_tr = loadTrainData(f='data/train_14x14.csv')
    m, n = X_tr.shape
    
    acc = []
    ls = np.linspace(0.2, 10, 20)
    for lmbd in ls:
        dig = DigitRecognizer(bits*bits, 27, 10)
        dig.setTrainParams(lmbd=lmbd, maxiter=250)
        
        #train
        dig.train(X_tr, y_tr)
        
        #check predictions
        pred = dig.predict(X_tr)
        #accuracy
        acc.append( (pred == y_tr.flatten()).sum()* 100.0 / m )
        print "Accuracy = %s%%"%acc[-1]

    plt.plot(ls, acc)
    plt.xlabel("Lambda")
    plt.ylabel("accuracy")
    plt.savefig('tuneLambda.png')
    plt.close()

def plotJhist():
    #load data
    X_tr, y_tr = loadTrainData(f='data/train_14x14.csv')
    m, n = X_tr.shape
    
    dig = DigitRecognizer(bits*bits, 27, 10)
    dig.setTrainParams(maxiter=250)
    
    #train
    dig.train(X_tr, y_tr)
    
    #check predictions
    pred = dig.predict(X_tr)
    #accuracy
    acc = (pred == y_tr.flatten()).sum()* 100.0 / m 
    print "Accuracy = %s%%"%acc

    plt.plot(dig.Jhist)
    plt.show()
    
def learningCurve():
    #load data
    X_tr, y_tr = loadTrainData(f='data/train_14x14.csv')
    m, n = X_tr.shape
    
    acc = []
    Jhist = []
    
    #incrementally by 100
    ls = [s for s in range(100, m, 100)]
    for s in ls:
        dig = DigitRecognizer(bits*bits, 27, 10)
        dig.setTrainParams(maxiter=100)
        
        print y_tr.shape
        #train with partial sample
        dig.train(X_tr[:s,:], y_tr[:s,])
        
        #check predictions against all samples
        pred = dig.predict(X_tr)
        #accuracy
        acc.append( (pred == y_tr.flatten()).sum()* 100.0 / m )
        Jhist.append(dig.J)
        print "Accuracy = %s%%"%acc[-1]
        print "Cost = %s"%dig.J
    
    plt.subplot(1, 2, 1).plot(ls, acc)
    plt.subplot(1, 2, 2).plot(ls, Jhist)
    plt.savefig("data/Learning_curve.png")
    
def trySeveral():
    """
        tries several times until reaching the best accuracy
        Since the initialization is random, there is hope
        that some will have higher fitting
    """
    #load data
    X_tr, y_tr = loadTrainData(f='data/train_14x14.csv')
    m, n = X_tr.shape
    times = 20
    bestDig = None
    bestAcc = 0
    l2 = 56#27
    
    for i in range(times):
        #optimal parameters so far

        dig = DigitRecognizer(bits*bits, l2, 10)
        dig.setTrainParams(maxiter=250, lmbd=0)
        
        #train
        dig.train(X_tr, y_tr)
        
        #check predictions
        pred = dig.predict(X_tr)
        #accuracy
        acc = (pred == y_tr.flatten()).sum()* 100.0 / m 
        print "Accuracy = %s%%"%acc
        #keep the best
        if acc > bestAcc:
            bestAcc = acc
            bestDig = dig
    
    print "Best so far = %s%%"%bestAcc
    #save it
    bestDig.save('data/theta%s_%s.txt'%(l2, bestAcc))
    
def analyzeFailures():
    #load best trained model
    l2 = 27
    dig = DigitRecognizer(bits*bits, l2, 10)
    best = "theta27_90.5688230672.txt"
    dig.load("data/%s"%best)
    #load data
    X_tr, y_tr = loadTrainData(f='data/train_14x14.csv')
    
    #make predictions
    pred = dig.predict(X_tr)
    #filter wrong cases
    wrong = (pred != y_tr.flatten())
    #hypothesis values for wrong
    X_w = X_tr[wrong,:]
    y_w = y_tr[wrong,:].flatten()
    h_w = dig.h[wrong,:]
    pred_w = pred[wrong]
    
    m = wrong.sum()
    print m, "wrong samples"
    #print a matrix
    print "correct label (rows) vs predicted label (colums)"
    print "  \\| "+("".join("%4s"%i for i in range(10)))
    print "_"*44
    for c in range(10):
        s = '%2s | ' %c
        for p in range(10):
            #count and filter
            x = np.logical_and(y_w == c , pred_w == p).sum()
            s += "%4s"%x
        print s
        
    """
     #pick 9 vs 4, some at random
    c = 9; p = 4
    ind = np.logical_and(y_w == c , pred_w == p)
    print "%s vs %s"%(c, p)
    h = h_w[ind,:]
    #for i in range(ind.sum()):
    #    print ''.join(' %.3f'%x for x in h[i,:])
    print "averages"
    print ''.join(' %.3f'%x for x in h.sum(0)/ind.sum())
    
    c = 3; p = 5
    ind = np.logical_and(y_w == c , pred_w == p)
    print "%s vs %s"%(c, p)
    h = h_w[ind,:]
    print "averages"
    print ''.join(' %.3f'%x for x in h.sum(0)/ind.sum())
    """
        
def analyzeFailingTheshold():
    #load best trained model
    l2 = 27
    dig = DigitRecognizer(bits*bits, l2, 10)
    best = "theta27_91.3619047619.txt"
    dig.load("data/%s"%best)
    #load data
    X_tr, y_tr = loadTrainData(f='data/train_14x14.csv')
    m = X_tr.shape[0]
    #make predictions
    pred = dig.predict(X_tr)

    #filter wright and wrong cases
    wrong = (pred != y_tr.flatten())
    right = (pred == y_tr.flatten())
    #hypothesis values for wrong
    h_w = dig.h[wrong]
    h_r = dig.h[right]
    

    print wrong.sum(), "wrong samples"
    print right.sum(), "right samples"
    acc = right.sum()* 1.0 / m
    
    #define a trheshold
    for thr in np.linspace(0.5,0.9, 5):
        #how many bad have max below threshold?
        wrongUnderT = (h_w.max(1) < thr).sum()
        #how many good have max below threshol?
        rightUnderT = (h_r.max(1) < thr).sum()
        print "Threshold > %s"%thr       
        print "wrong we will look: %s (%s%%)"%(wrongUnderT, wrongUnderT*100.0/wrong.sum())
        print "right we will look: %s (%s%%)"%(rightUnderT, rightUnderT*100.0/right.sum())
        
        #if we only gave answer to those over the threshold
        #how good will be the accuracy?
        #how many are the good ones over threshold
        wrongOverT = (h_w.max(1) >= thr).sum()
        #bad ones over threshold
        rightOverT = (h_r.max(1) >= thr).sum()
        acc_t = rightOverT*100.0 / (rightOverT + wrongOverT)
        print "accuracy of the remaining samples: %s%%"%acc_t
        print "samples remaining: %s%%"% (100.0*(rightOverT+wrongOverT)/m)
    
    
def main():
    #tuneL2()
    #tuneEpsilon()
    #tuneLambda()
    trySeveral()
    #analyzeFailures()
    #analyzeFailingTheshold()

if __name__ == '__main__':
    main()
    




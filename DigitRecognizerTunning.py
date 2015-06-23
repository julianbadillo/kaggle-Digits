#!/usr/bin/python
"""
    A class for Recognizing Digits
"""

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
    plt.savefig('tuneL2.png')
    plt.close()

def tuneEpsilon():
    #load data
    X_tr, y_tr = loadTrainData(f='data/train_14x14.csv')
    
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
    plt.savefig("Learning_curve.png")
    
def trySeveral():
    """
        tries several times until reaching the best accuracy
        Since the initialization is random, there is hope
        that some will have higher fitting
    """
    #load data
    X_tr, y_tr = loadTrainData(f='data/train_14x14.csv')
    
    times = 20
    bestDig = None
    bestAcc = 0
    l2 = 27
    
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
    bestDig.save('theta%s_%s.txt'%(l2, bestAcc))
    
def main():
    #tuneL2()
    #tuneEpsilon()
    #tuneLambda()
    trySeveral()

if __name__ == '__main__':
    main()
    




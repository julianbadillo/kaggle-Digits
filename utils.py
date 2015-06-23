#!/usr/bin/python
"""
    Utilities and formulas
"""
import numpy as np
import scipy as sp
import random
from math import sqrt, ceil

def loadTrainData(f='data/train.csv', debug=False):
    """
    Loads the Training data
    """
    #read the train set
    data = np.loadtxt(f, delimiter=',', skiprows=1)
    if debug:
        print "Shape of the Train set", data.shape
    m, n = data.shape
    y_tr = data[:,:1]
    X_tr = data[:,1:]
    return X_tr, y_tr
    
def loadTestData(f='data/test.csv', debug=False):
    #read the test set
    data = np.loadtxt(f, delimiter=',', skiprows=1)
    if debug:
        print "Shape of the Test set", data.shape
    return data

def initMatrix(r, c, e):
    """
    Creates a matrix [r x c] with random
    values between (-e,e)
    """
    m = np.random.rand(r,c)
    m = -e + m*2*e
    return m

def sigmoid(x):
    """
    logistic function
    """
    return 1/(1+np.exp(-x))

def sigmoidGrad(x):
    """
    Derivative of sigmoid
    """
    g = sigmoid(x)
    return g*(1-g)

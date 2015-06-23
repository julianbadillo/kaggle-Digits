#!/usr/bin/python
"""
    Show plots for understanding the data
"""
import numpy as np
import scipy as sp
import scipy.io as sio
from scipy.optimize import fmin_cg
from matplotlib import pyplot as plt
import random
from math import sqrt, ceil
from utils import *

def showExample(X, y, index, bits = 20):
    """
        Plots selected examples
    """
    n = ceil(sqrt( len(index)))
    j = 0
    for i in index:
        x = -np.reshape(X[i,:], (bits,bits))
        #show the digit
        #show the image
        subplot = plt.subplot(n, n, j+1)
        subplot.set_title("%s"%int(y[i]))
        im = subplot.imshow(x)
        im.axes.get_xaxis().set_visible(False)
        im.axes.get_yaxis().set_visible(False)
        im.set_cmap('gray')
        j += 1
    plt.show()
    

def showDistribution(y):
    """
        sample distribution
    """
    x = np.unique(y)
    f = np.zeros(x.shape)
    for i in range(x.size):
        f[i] = (y == x[i]).sum()
    plt.bar(x, f)
    plt.show()

def showPartialDistributions(y):
    """
        Partial distributions, to plot the learning curve.
    """
    j = 0
    #incrementally take 100, 200, 300...
    for s in range(100, len(y)+1, 100):
        yp = y[:s] #first s samples
        xp = np.unique(yp)
        f = np.zeros(xp.shape)
        #frequency
        for i in range(xp.size):
            f[i] = (yp == xp[i]).sum()
            
        subplot = plt.subplot(4, 3, j+1)
        subplot.set_title("%s samples"%s)
        subplot.bar(xp, f)
        
        j += 1

    #plt.show()
    plt.savefig('data/distributions.png')
    

def main():    
    bits = 28
                              
    X_tr, y_tr = loadTrainData()
               
    #how many of each
    showPartialDistributions(y_tr)
    
    #show random sample
    samples = np.random.permutation(m)[:49]
    showExample(X_tr, y_tr, samples, bits)         
    
if __name__ == '__main__':
    main()

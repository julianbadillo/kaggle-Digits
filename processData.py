#!/usr/bin/python
"""
    Initial processing to get smaller ammount of features
"""
import numpy as np
from math import sqrt, ceil
from utils import *

def transformData(X):
    """
        Changes from 28x28 bits matrix
        To 14x14 bit matrix
        this improves the efficiency of the algorithms
    """
    #from 28 bits to 14 bits
    bits1 = 28
    bits2 = 14
    sq = 2
    m = X.shape[0]
    Xnew = np.zeros((m, bits2*bits2))
    for i in range(m):
        #reshape and process
        mat = np.reshape(X[i], (bits1,bits1))
        matnew = np.zeros((bits2,bits2))
        #take sqxsq pixels
        for j in range(bits2):
            for k in range(bits2):
                matnew[j,k] = np.average(mat[sq*j:sq*j+2,sq*k:sq*k+2])
        Xnew[i] = matnew.flatten()
    return Xnew

def transformData2(X):
    """
        Transforms to 14x14 binary matrix.
        Looses a lot of information.
    """
    #from 28 bits to 14 bits
    bits1 = 28
    bits2 = 14
    sq = 2
    m = X.shape[0]
    Xnew = np.zeros((m, bits2*bits2))
    for i in range(m):
        #reshape and process
        mat = np.reshape(X[i], (bits1,bits1))
        matnew = np.zeros((bits2,bits2))
        #take sqxsq pixels
        for j in range(bits2):
            for k in range(bits2):
                a = np.average(mat[sq*j:sq*j+2,sq*k:sq*k+2])
                matnew[j,k] = 1.0 if a >= 128 else 0
        Xnew[i] = matnew.flatten()
    return Xnew

def main():    
    bits = 14
                              
    #Transform the train set
    X, y = loadTrainData()
    X2 = transformData(X)
    #insert y
    data = np.insert(X2, 0, y.flatten(), axis=1)
    np.savetxt('train_14x14.csv',data , fmt='%s' , delimiter=',', header='pixels')
    
    #transform the test set
    X = loadTestData()
    X2 = transformData(X)
    np.savetxt('test_14x14.csv',X2 , fmt='%s' , delimiter=',', header='pixels')
    
if __name__ == '__main__':
    main()

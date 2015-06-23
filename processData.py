#!/usr/bin/python
"""
    Show plots for understanding the data
"""
import numpy as np
from math import sqrt, ceil

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
                              
    #read the train set
    data = np.loadtxt("train_short.csv", delimiter=',', skiprows=1)
    print "Shape of the Train set", data.shape
    m, n = data.shape
    y = data[:,:1]
    X = data[:,1:]

    X2 = transformData(X)
               
    #show random sample
    samples = np.random.permutation(m)[:49]
    #showExample(X2, y, samples, bits)
    #add y column and save  
    data2 = np.insert(X2, 0, y.flatten(), axis=1)
    #data2 = X2
    print "Shape of the Train set", data2.shape
    np.savetxt('train_short14x14.csv',data2 , fmt='%s' , delimiter=',', header='pixels')      
    
if __name__ == '__main__':
    main()

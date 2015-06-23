#!/usr/bin/python
"""
    An example of a neural network
"""

import numpy as np
import scipy as sp
import scipy.io as sio
from scipy.optimize import fmin_cg
from matplotlib import pyplot as plt
import random
from math import sqrt, ceil

def showExample(X,y, index, bits = 20):
    n = ceil(sqrt( len(index)))
    j = 0
    for i in index:
        x1 = np.reshape(X[i,:], (bits,bits)).T
        y1 = y[i]
        #show the digit
        print "Digit:", y1
        #show the image
        subplot = plt.subplot( n, n, j+1)
        im = subplot.imshow(x1)
        im.set_cmap('gray')
        j += 1
    plt.show()
    
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

def calculatePrediction(X, Theta1, Theta2):
    """
    For all samples
    """
    m = X.shape[0] #number of examples, pixels = n1
    #insert a column of ones
    a1 = np.insert(X, 0, np.ones(m), axis=1)
    #compute layer 2
    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2) #one row per sample, one column per unit
    
    #add column of ones
    a2 = np.insert(a2, 0, np.ones(m), axis=1)
    
    #computing layer 3
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)
    
    #maximun index by rows
    p = np.argmax(a3, axis=1)
    #shift to fit the labels
    p += 1
    return p

def cost(Theta, X, y, n1, n2, n3, lmd):
    """
    Wrapper
    """
    global Theta_grad
    #reshape matrix
    Theta1 = Theta[:n2*(n1+1)].reshape((n2,n1+1))
    Theta2 = Theta[n2*(n1+1):].reshape((n3,n2+1))
    
    J, Theta1_grad, Theta2_grad = calculateCost(X, y, Theta1, Theta2,
                                                 n1, n2, n3, lmd)
    #concat gradient
    Theta_grad = np.concatenate((Theta1_grad.flatten(), 
                                Theta2_grad.flatten()))
    print "J = ", J
    return J

def grad(Theta, X, y, n1, n2, n3, lmd):
    """
    Wrapper
    """
    global Theta_grad
    
    if Theta_grad == None:
        cost(Theta, X, y, n1, n2, n3, lmd)
    return Theta_grad


def calculateCost(X, y, Theta1, Theta2, n1, n2, n3, lmd):
    """
    Calculates the cost of h
    """
    m = X.shape[0] #number of examples, pixels = n1
    #insert a column of ones
    a1 = np.insert(X, 0, np.ones(m), axis=1)
    #compute layer 2
    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2) #one row per sample, one column per unit
    
    #add column of ones
    a2 = np.insert(a2, 0, np.ones(m), axis=1)
    
    #computing layer 3
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)
    
    #diff and cost of logistic regression
    J = np.sum(-y*np.log(a3) - (1-y)*np.log(1-a3))/m
    
    #regularization
    J += 0.5*lmd/m*( np.sum(Theta1[:,1:]*Theta1[:,1:]) +
                     np.sum(Theta2[:,1:]*Theta2[:,1:]) )
    
    delta1 = np.zeros(Theta1.shape)
    delta2 = np.zeros(Theta2.shape)
    
    #each sample
    for t in range(m):
        #precalculated rows -> forward propagation
        a_1 = a1[t:t+1,:].T
        z_2 = z2[t:t+1,:].T
        a_2 = a2[t:t+1,:].T
        a_3 = a3[t:t+1,:].T
        
        #error on last layer
        d_3 = a_3 - y[t:t+1,:].T
        #dark magic with gradients back propagation
        d_2 = np.dot(Theta2.T, d_3)
        d_2 = d_2[1:] * sigmoidGrad(z_2) #column vector, one row per unit
        delta1 += np.dot(d_2, a_1.T) #%delta matrix for acumulation removing the extra column
        delta2 += np.dot(d_3, a_2.T) 
    
    Theta1_grad = delta1/m
    Theta2_grad = delta2/m
    
    #regularization terms to gradient
    Theta1_grad[:,1:] += float(lmd)/m*Theta1[:,1:]
    Theta2_grad[:,1:] += float(lmd)/m*Theta2[:,1:]

    return J, Theta1_grad, Theta2_grad


def gradientCheck(X, y, Theta, n1, n2, n3, lmd):
    """
    Checks the gradient calculted by back propagation
    with a linear aproximation
    """
    s = 0
    d = grad(Theta, X, y, n1, n2, n3, lmd)
    epsilon = 0.001
    #check 30 weights at random
    tries = 30 
    for i in range(tries):
        r = random.randrange(Theta.size)
        Theta[r] += epsilon
        f2 = cost(Theta, X, y, n1, n2, n3, lmd)
        Theta[r] -= 2*epsilon
        f3 = cost(Theta, X, y, n1, n2, n3, lmd)
        #derivative aproximation
        fp = (f2 - f3)/2/epsilon
        print "Grad",d[r]
        print "Numeric ",fp
        print "Grad/Numeric",d[r]/fp
        s += abs(d[r]-fp)
        Theta[r] += epsilon
    
    print "Average difference:",s/tries


def old_main():
    ## Setup the parameters you will use for this exercise
    input_layer_size  = 400;  # 20x20 Input Images of Digits
    hidden_layer_size = 25;   # 25 hidden units
    num_labels = 10;          # 10 labels, from 1 to 10   
                              # (note that we have mapped "0" to label 10)


    data = sio.loadmat('ex4data1.mat')
    X, y = data['X'], data['y']
    m, n = X.shape
    #transform Y to a feature-based matrix
    #since the params are from 1 to 10, I moved one back
    y2 = y.flatten()
    y = np.zeros((m, num_labels))
    for i in range(m):
        y[i,y2[i]-1] = 1
    
    #show random sample
    samples = np.random.permutation(m)[:16]
    #showExample(X,y2,samples)
    
    # loading pre-initialized neural network
    data = sio.loadmat('ex4weights.mat')
    print data.keys()
    Theta1, Theta2 = data['Theta1'], data['Theta2']
    Theta = np.concatenate((Theta1.flatten(), Theta2.flatten()))
    lambd = 0
    
    #check that the calculations are correct
    J, Theta2_grad, Theta2_grad = calculateCost(X, y, Theta1, Theta2, input_layer_size,
            hidden_layer_size, num_labels, lambd)
    print "Cost of loaded. should be around 0.287629"
    print J
    
    lambd = 1
    J, Theta2_grad, Theta2_grad = calculateCost(X, y, Theta1, Theta2, input_layer_size,
            hidden_layer_size, num_labels, lambd)
    print "Cost of loaded. should be around 0.383770"
    print J
    
    lambd = 3
    J, Theta2_grad, Theta2_grad = calculateCost(X, y, Theta1, Theta2, input_layer_size,
            hidden_layer_size, num_labels, lambd)
    
    print "Cost of loaded. should be around 0.576051"
    print J
    
    print "Gradient check with linear aproximation"
    global Theta_grad
    Theta_grad = None
    lambd = 3
    
    #gradientCheck(X, y, Theta, input_layer_size,
    #        hidden_layer_size, num_labels, lambd)
    pred = calculatePrediction(X, Theta1, Theta2)
    acc = (pred == y2).sum()* 100.0 / m
    print "Accuracy of preloaded weights %s%%"%acc
    
    #make predictions and calculate accuracy
    pred = calculatePrediction(X, Theta1, Theta2)
    samples = np.random.permutation(m)[:16]
    print ' '.join(str(pred[i]) for i in samples)
    acc = (pred == y2).sum()* 100.0 / m
    print "Accuracy of %s%%"%acc
    showExample(X, y2, samples)
    
    #failing samples
    fails = (pred != y2)
    print "%s failing samples "%fails.sum()
    print ' '.join(str(p) for p in pred[fails])
    fails = np.array(range(m))[fails]
    showExample(X, y2, fails)
    




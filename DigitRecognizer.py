#!/usr/bin/python
"""
    A class for Recognizing Digits
"""

import numpy as np
from scipy.optimize import fmin_cg
import random
from math import sqrt, ceil
from utils import *

class DigitRecognizer:
    """
    Uses a 3 layer neural network
    """
    
    def __init__(self, l1, l2, l3 = 10):
        """
        Initializes the recognizer
        l1 = input layer size (pixels on image)
        l2 = hidden layer size
        l3 = output layer size (labels) digits
        """
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
    
    def setTrainParams(self, **params):
        """
        Training params for tweaking
        """
        #range for random theta initialization
        self.epsilon = params['epsilon'] if 'epsilon' in params else 0.35 
        #regularization weight
        self.lmbd = params['lmbd'] if 'lmbd' in params else 1
        #maximun iterations
        self.maxiter = params['maxiter'] if 'maxiter' in params else 100
        
    def train(self, X, y):
        """
        Performs network layer training
        """
        self.m = X.shape[0] #number of examples, pixels = self.l1
        
        self.Theta1 = initMatrix(self.l2, self.l1+1, self.epsilon)
        self.Theta2 = initMatrix(self.l3, self.l2+1, self.epsilon)
        self.Theta = np.concatenate((self.Theta1.flatten(), self.Theta2.flatten()))
        
        #redimension y to fit multi-class logistic function
        y2 = np.zeros((self.m, self.l3))
        for i in range(self.m):
            y2[i][int(y[i])] = 1
        y = y2
        
        #minimization function
        self.Theta_grad = None
        self.ite = 0
        self.Jhist = []
        print "Calculating optimal parameters"
        ThOpt = fmin_cg(self.cost, self.Theta, fprime = self.grad,
                        args=(X, y),
                        maxiter= self.maxiter, disp=False)
        print "Finished calculating"
        
        #unwrap (probably uneeded)
        #self.Theta1 = ThOpt[:self.l2*(self.l1+1)].reshape((self.l2,self.l1+1))
        #self.Theta2 = ThOpt[self.l2*(self.l1+1):].reshape((self.l3,self.l2+1))
        #TODO train the second identifier for threshold
        #what about picking the samples with the highest confidence
        #and do a K-distance algorithm
        
    def cost(self, Theta, X, y):
        """
        Wrapper for cost
        """
        #cached value
        if self.J is not None and (self.Theta == Theta).all():
            #print "cached J"
            self.Jhist.append(self.J)
            return self.J
        #update
        self.Theta = Theta
        
        #reshape matrix
        self.Theta1 = Theta[:self.l2*(self.l1+1)].reshape((self.l2,self.l1+1))
        self.Theta2 = Theta[self.l2*(self.l1+1):].reshape((self.l3,self.l2+1))
        
        self.__calculateCost(X, y)
        
        #concat gradient
        self.Theta_grad = np.concatenate((self.Theta1_grad.flatten(), 
                                    self.Theta2_grad.flatten()))
        self.Jhist.append(self.J)
        return self.J
    
    def grad(self, Theta, X, y):
        """
        Wrapper for gradient
        """
        #cached value
        if self.Theta_grad is not None and (self.Theta == Theta).all():
            #print "cached grad"
            return self.Theta_grad
        #update
        self.Theta = Theta
        
        #reshape matrix
        self.Theta1 = Theta[:self.l2*(self.l1+1)].reshape((self.l2,self.l1+1))
        self.Theta2 = Theta[self.l2*(self.l1+1):].reshape((self.l3,self.l2+1))
        
        self.__calculateCost(X, y)
        
        #concat gradient
        self.Theta_grad = np.concatenate((self.Theta1_grad.flatten(), 
                                    self.Theta2_grad.flatten()))
        return self.Theta_grad
        
    def __calculateCost(self, X, y):
        """
        Calculates both cost and gradient in one go since it's much quicker
        """
        #insert a column of ones
        a1 = np.insert(X, 0, np.ones(self.m), axis=1)
        #compute layer 2
        z2 = np.dot(a1, self.Theta1.T)
        a2 = sigmoid(z2) #one row per sample, one column per unit
        
        #add column of ones
        a2 = np.insert(a2, 0, np.ones(self.m), axis=1)
        
        #computing layer 3
        z3 = np.dot(a2, self.Theta2.T)
        a3 = sigmoid(z3)
        
        #diff and cost of logistic regression
        self.J = np.sum(-y*np.log(a3) - (1-y)*np.log(1-a3))/self.m
        
        #regularization
        self.J += 0.5*self.lmbd/self.m*( np.sum(self.Theta1[:,1:]*self.Theta1[:,1:]) +
                         np.sum(self.Theta2[:,1:]*self.Theta2[:,1:]) )
        
        delta1 = np.zeros(self.Theta1.shape)
        delta2 = np.zeros(self.Theta2.shape)
        
        #each sample
        for t in range(self.m):
            #precalculated rows -> forward propagation
            a_1 = a1[t:t+1,:].T
            z_2 = z2[t:t+1,:].T
            a_2 = a2[t:t+1,:].T
            a_3 = a3[t:t+1,:].T
            
            #error on last layer
            d_3 = a_3 - y[t:t+1,:].T
            #dark magic with gradients back propagation
            d_2 = np.dot(self.Theta2.T, d_3)
            d_2 = d_2[1:] * sigmoidGrad(z_2) #column vector, one row per unit
            delta1 += np.dot(d_2, a_1.T) #%delta matrix for acumulation removing the extra column
            delta2 += np.dot(d_3, a_2.T) 
        
        self.Theta1_grad = delta1/self.m
        self.Theta2_grad = delta2/self.m
        
        #regularization terms to gradient
        self.Theta1_grad[:,1:] += float(self.lmbd)/self.m*self.Theta1[:,1:]
        self.Theta2_grad[:,1:] += float(self.lmbd)/self.m*self.Theta2[:,1:]
        self.ite += 1
    
    def predict(self, X):
        """
        Calculates predictions according to trained parameters
        """
        m = X.shape[0] #number of examples

        #insert a column of ones
        a1 = np.insert(X, 0, np.ones(m), axis=1)
        
        #compute layer 2
        z2 = np.dot(a1, self.Theta1.T)
        a2 = sigmoid(z2) #one row per sample, one column per unit
        
        #add column of ones
        a2 = np.insert(a2, 0, np.ones(m), axis=1)
        
        #computing layer 3
        z3 = np.dot(a2, self.Theta2.T)
        a3 = sigmoid(z3)
        
        #maximun index by rows
        p = np.argmax(a3, axis=1)
        #TODO fix a threshold , so if the max doesn't pass that then feed to another algorithm.
        #TODO deciding between best and second with another identifier
        
        #save the latest hypothesis value
        self.h = a3
        return p    
    
    def save(self, f):
        """
            Saves to the given file name
            This should be called after training
            and testing
        """
        self.Theta = np.concatenate((self.Theta1.flatten(), self.Theta2.flatten()))
        np.savetxt(f, self.Theta)
        
    def load(self, f):
        """
            Loads a previously saved model
            ready to make predictions
        """
        self.Theta = np.loadtxt(f)
        #reshape matrix
        self.Theta1 = self.Theta[:self.l2*(self.l1+1)].reshape((self.l2,self.l1+1))
        self.Theta2 = self.Theta[self.l2*(self.l1+1):].reshape((self.l3,self.l2+1))
        

def gradientCheck(X, y, Theta, l1, l2, l3, lmbd):
    """
    Checks the gradient calculted by back propagation
    with a linear aproximation
    """
    s = 0
    d = grad(Theta, X, y, self.l1, self.l2, self.l3, lmbd)
    epsilon = 0.001
    #check 30 weights at random
    tries = 30 
    for i in range(tries):
        r = random.randrange(Theta.size)
        Theta[r] += epsilon
        f2 = cost(Theta, X, y, self.l1, self.l2, self.l3, lmbd)
        Theta[r] -= 2*epsilon
        f3 = cost(Theta, X, y, self.l1, self.l2, self.l3, lmbd)
        #derivative aproximation
        fp = (f2 - f3)/2/epsilon
        print "Grad",d[r]
        print "Numeric ",fp
        print "Grad/Numeric",d[r]/fp
        s += abs(d[r]-fp)
        Theta[r] += epsilon
    
    print "Average difference:",s/tries


def main():
    pass

if __name__ == '__main__':
    main()
    




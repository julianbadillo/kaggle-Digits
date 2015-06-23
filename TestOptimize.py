#!/usr/bin/python
"""
    For testing optimization
"""

import numpy as np
import scipy as sp
import scipy.io as sio
from scipy.optimize import *
from matplotlib import pyplot as plt
import random
from math import sqrt, ceil


class Model:
    def __init__(self):
        self.e = 2.0
    
    def train(self, X, y):
        #features and examples
        self.m, self.n = X.shape
        #add a column of ones

        X = np.insert( X, 0, np.ones(self.m), axis=1)
        #hyp
        self.Theta = np.zeros(self.n + 1)
        self.Theta_grad = None
        self.J = None
        fmin_cg(self.cost, self.Theta, fprime=self.grad,
                        args=(X, y),
                        maxiter=200, disp=True)

    def cost(self, Theta, X, y):
        """
        Cost Function
        """
        #if theta not changed
        if self.J and (self.Theta == Theta).all():
            print "J already calculated"
            return self.J
        #update theta
        self.Theta = Theta
        self.__calculateCostAndGrad(X, y)
        return self.J
        
    def grad(self, Theta, X, y):
        """
        Gradient function
        """
        #if theta not changed
        if self.Theta_grad is not None and (self.Theta == Theta).all():
            print "Grad already calculated"
            return self.Theta_grad
        #update theta
        self.Theta = Theta
        self.__calculateCostAndGrad(X, y)
        return self.Theta_grad
    
    def __calculateCostAndGrad(self, X, y):
        """
            Calculates cost and gradient, since
            it's more efficient in a single go
        """
         #square difference vs. predicted and actual
        h = np.dot(self.Theta, X.T**2)
        d = h - y
        sq = np.power(d, 2)
        self.J = np.average(sq) / 2.0
        #gradient = diff*X^2, sum by columns
        self.Theta_grad = np.average(d*(X.T**2), 1)
                

def main():
    X = np.zeros((400,2))
    x1 = np.linspace(0,100, 20)
    x2 = np.linspace(0,100, 20)
    x1, x2 =  np.meshgrid(x1, x2)
    y = 3.2*x1**2 - 1.4*x2**2 + 3
    X[:,0] = x1.flatten()
    X[:,1] = x2.flatten()
    y = y.flatten()
    #add random noise
    eps = 0.5
    y += 2*eps*np.random.random(y.shape) - eps
    model = Model()
    model.train(X, y)   
    print model.Theta
    
    
if __name__ == '__main__':
    main()
    

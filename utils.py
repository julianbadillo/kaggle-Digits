#!/usr/bin/python
"""
    Utilities and formulas
"""
import numpy as np
import scipy as sp
import random
from math import sqrt, ceil

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

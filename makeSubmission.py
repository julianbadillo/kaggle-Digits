#!/usr/bin/python
"""
    Makes the submition output with a trained models
"""

import numpy as np
from DigitRecognizer import DigitRecognizer
from processData import transformData

bits = 14


def save(pred):
    f = open('data/submission.csv', 'w')
    #header
    f.write('"ImageId","Label"\n')
    for i in range(pred.size):
        f.write('%s,"%s"\n'%(i+1,pred[i]))
    f.close()
    
    
def main():
    #TODO do with loaded model. load data for training
    print "Loading trained model"
    dig = DigitRecognizer(bits*bits, 27, 10)
    #load best trained so far
    best = "theta27_90.5688230672.txt"
    dig.load("data/%s"%best)
    print best, "loaded"
    
    #test
    test = np.loadtxt("data/test_14x14.csv", delimiter=',', skiprows=1)
    X = test[:,:]
    print X.shape, test.shape
    #transfor the data to fit the new bit size
    #X = transformData(X)
    print "making predictions"
    pred = dig.predict(X)
    print "finished", pred.shape
    #predictions to submission file
    save(pred)
    
    
if __name__ == '__main__':
    main()
    




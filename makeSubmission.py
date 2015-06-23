#!/usr/bin/python
"""
    Makes the submition output with a trained models
"""

import numpy as np
from DigitRecognizer import DigitRecognizer
from processData import transformData

bits = 14


def save(pred):
    f = open('predictions.csv', 'w')
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
    dig.load("theta27_90.12.txt")

    #test
    print "making predictions"
    test = np.loadtxt("test_14x14.csv", delimiter=',', skiprows=1)
    X = test[:,:]
    #transfor the data to fit the new bit size
    #X = transformData(X)
    
    pred = dig.predict(X)
    
    #predictions to submission file
    save(pred)
    
    
if __name__ == '__main__':
    main()
    




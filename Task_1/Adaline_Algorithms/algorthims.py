# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 09:52:36 2023

@author: fat7i nasser
"""
import numpy as np
l=0.001
mse=0.01
def fitmode(x,y,w):
    i=1
    
    for c in range(0,x.shape[0]):
        y_hat=np.dot(w.reshape(1,-1),x[c].reshape(-1,1))
        error=y[c]-y_hat
        terror=error**2+0
        w=w+(l*error*x[c])
        if(c==(i*(x.shape[1]-1))):
            i+=1
            mses=(1/2)*(terror/x.shape[1])
            terror=0
        if(mses<=mse):
            break;
            
        return w    


#def splitdata(data):
    

samples=0
trainSamples=0
def testmodel(w,y,x):
    correct=0;
    wrong=0;
    for c in range(trainSamples, x.shape[0]):
        y_hat=np.dot(w.reshape(1,-1),x[c].reshape(-1,1))
        if(y[c]==y_hat):
            correct+=1
        else:
            wrong+=1
            
    return correct
            
    


def split_data(start, all_data):
    return (all_data.loc[start: start + samples - 1, all_data.columns].
            sample(frac=1, random_state=42)).reset_index(drop=True)

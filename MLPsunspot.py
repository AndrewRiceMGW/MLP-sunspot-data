#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:56:40 2019

@author: andrew
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

data = pd.read_csv("sunspot.csv", header = None)

year = data.iloc[:, 0]
relNums = data.iloc[:, 1]
year = year.to_numpy()
relNums = relNums.to_numpy()
nrmY = relNums
ymin = np.amin(nrmY)
ymax = np.amax(nrmY)
relNums = 2.0* ((nrmY - ymin) / (ymax - ymin) -0.5) 
Ss = np.transpose(relNums)
idim=10
odim=len(Ss)-idim

y =[]
for i in range(0, odim, 1):
    y.append(Ss[i+idim])
    
y = np.asarray(y)

x = np.zeros((278, 10))    
for i in range(0, odim, 1):
    for j in range(0,idim, 1):
        x[i][j] = Ss[i -j + idim]
 # Problem here for some reason x(2,1) is switched with x(1,1)
Patterns = np.transpose(x)
NINPUTS = idim; NPATS = odim; NOUTPUTS = 1; NP = odim;
Desired = y; NHIDDENS = 5; prnout=Desired;
LR = 0.001; Momentum = 0; DerivIncr = 0; deltaW1 = 0; deltaW2 = 0;
Inputs1 = np.concatenate((Patterns, np.ones((1, NPATS))))
Weights1 = 0.5 * (np.random.rand(NHIDDENS, 1+NINPUTS)-0.5)
Weights2 = 0.5 * (np.random.rand(1, 1+NHIDDENS)-0.5)
TSS_Limit = 0.02;
epochs = 100
for epoch in range(1, epochs+1, 1):
    # Feedforward
    NetIn1 = np.dot(Weights1, Inputs1)
    Hidden=np.divide(1-2, (np.exp(2*NetIn1)+1));
    Inputs2 = np.concatenate((Hidden, np.ones((1,NPATS))))
    NetIn2 = np.dot(Weights2, Inputs2)
    Out = NetIn2;  prnout=Out;
    Error = Desired - Out
    TSS = sum(sum((Error**2)))
    # Backpropogation
    Beta = Error
    bperr = np.dot(np.transpose(Weights2), Beta );
    HiddenBeta = np.multiply((1.0 - Hidden**2),  bperr[0:-1,:]);
    dW2 = np.dot(Beta, np.transpose(Inputs2))
    dW1 = np.dot(HiddenBeta, np.transpose(Inputs1))
    deltaW2 = LR * dW2 + Momentum * deltaW2
    deltaW1 = LR * dW1 + Momentum * deltaW1
    Weights2 = Weights2 + deltaW2
    Weights1 = Weights1 + deltaW1
    print("Epoch:", epoch, "Error:", TSS )
    
    
prnout = np.transpose(prnout)
plt.plot(year[10:], Desired, label = 'Desired')
plt.plot(year[10:], prnout[:], label = 'Actual')
plt.xlabel('Year')
plt.ylabel('sunspots')


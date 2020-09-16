#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import random
import pandas as pd
import time

n = 1000
m = 500

X= np.load('X2_save.npy')
A= np.load('A2_save.npy')

X1 = np.random.randn(m)
X2 = np.random.randn(m)
X3 = np.random.randn(m)

y1 = np.ones(m)
y2 = np.ones(m)
y3 = np.ones(m)

z1 = np.random.randn(m)
z2 = np.random.randn(m)
z3 = np.random.randn(m)

G1_1 = np.random.randn(m)
G2_1 = np.random.randn(m)
G3_1 = np.random.randn(m)

y1_1 = np.random.randn(m)
y2_1 = np.random.randn(m)
y3_1 = np.random.randn(m)

rho = 0.0001

A1 = A[0:300,:]
A2 = A[300:700,:]
A3 = A[700:n,:]

Y = A@X
Y1 = Y[0:300]
Y2 = Y[300:700]
Y3 = Y[700:n]

X_opt = np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T), Y)
ttime = 0

def dJ_dw(A,X,Y):
    return 2*np.dot(A.T,(np.dot(A,X)-Y))


# In[15]:


print("|| Iteration  |   primal residual    |     time    ||")

for i in range(1001):  
    start = time.time()

    z1 = X1/y1
    z2 = X2/y2
    z3 = X3/y3
    
    G1 = dJ_dw(A1,z1,Y1)
    G2 = dJ_dw(A2,z2,Y2)
    G3 = dJ_dw(A3,z3,Y3)
    
    X1_1 = (2/3) * X1 + (1/3) * X2 - rho * G1
    X2_1 = (1/3) * X1 + (1/3) * X2 + (1/3) * X3 - rho * G2
    X3_1 = (1/3) * X2 + (2/3) * X3 - rho * G3
    
    y1_1 = (2/3) * y1 + (1/3) * y2 
    y2_1 = (1/3) * y1 + (1/3) * y2 + (1/3) * y3
    y3_1 = (1/3) * y2 + (2/3) * y3
            
    X1 = X1_1
    X2 = X2_1
    X3 = X3_1
    
    y1 = y1_1
    y2 = y2_1
    y3 = y3_1
    
    result1 = np.linalg.norm(X_opt-X1,2)
    result2 = np.linalg.norm(X_opt-X2,2)
    result3 = np.linalg.norm(X_opt-X3,2)
    
    end = time.time()
    ttime += end-start
    
    r = (1/3) * (result1 + result2 + result3)
    if (i % 100 == 0):
        print('||%10d  |%17.10f     |%12.5f ||' %(i, r, time_old + ttime))


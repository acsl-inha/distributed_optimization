#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

y1 = np.random.randn(m)
y2 = np.random.randn(m)
y3 = np.random.randn(m)

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


# In[2]:


import time

k=0

start = time.time()
ttime = 0
G1 = dJ_dw(A1,X1,Y1)
G2 = dJ_dw(A2,X2,Y2)
G3 = dJ_dw(A3,X3,Y3)

X1_1 = (2/3) * X1 + (1/3) * X2 - rho * G1
X2_1 = (1/3) * X1 + (1/3) * X2 + (1/3) *X3  - rho * G2
X3_1 = (1/3) * X2 + (2/3) * X3 - rho * G3

end = time.time()

time_old = end - start


# In[3]:


print("|| Iteration  |   primal residual    |     time    ||")
for i in range(1001):  
    start = time.time()
    
    G1_2 = dJ_dw(A1,X1_1,Y1)
    G2_2 = dJ_dw(A2,X2_1,Y2)
    G3_2 = dJ_dw(A3,X3_1,Y3)

    X1_2 = X1_1 + (2/3) * X1_1 + (1/3) * X2_1 -((5/6) * X1 + (1/6) * X2) - rho * (G1_2 - G1)
    X2_2 = (X2_1 + (1/3) * X1_1 + (1/3) * X2_1 + (1/3) * X3_1 - ((1/6) * X1 + (4/6) * X2 + (1/6) * X3)  
            - rho *(G2_2-G2))
    X3_2 = X3_1 + (1/3) * X2_1 + (2/3) * X3_1 - ((1/6) * X2 + (5/6) * X3) - rho * (G3_2-G3)
    
    G1 = G1_2
    G2 = G2_2
    G3 = G3_2
    
    X1 = X1_1
    X2 = X2_1
    X3 = X3_1
            
    X1_1 = X1_2
    X2_1 = X2_2
    X3_1 = X3_2
    
    result1 = np.linalg.norm(X_opt-X1,2)
    result2 = np.linalg.norm(X_opt-X2,2)
    result3 = np.linalg.norm(X_opt-X3,2)
    
    end = time.time()
    ttime += end-start
    
    r = (1/3) * (result1 + result2 + result3)
    if (i % 100 == 0):
        print('||%10d  |%17.10f     |%12.5f ||' %(i, r, time_old + ttime))


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jax.numpy as jnp
import numpy as np
from jax import grad
from jax.ops import index_update, index
import jax
import random
import pandas as pd
import matplotlib.pyplot as plt
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


def dJ_dw(A,X,Y):
    return 2*np.dot(A.T,(np.dot(A,X)-Y))


# In[2]:


import time

arr1 = []
arr2 = []
arr3 = []

k=0
ttime = 0

y1 = dJ_dw(A1,X1,Y1)
y2 = dJ_dw(A2,X2,Y2)
y3 = dJ_dw(A3,X3,Y3)
subG10 = y1
subG20 = y2
subG30 = y3

print("|| Iteration  |   primal residual    |     time    ||")
for i in range(1001):
    start = time.time()

    
    X_1 = (2/3) * X1 + (1/3) * X2 - rho * y1
    X_2 = (1/3) * X1 + (1/3) * X2 + (1/3) *X3 - rho * y2
    X_3 = (1/3) * X2 + (2/3) * X3 - rho * y3
    
    subG1 = dJ_dw(A1,X_1,Y1)
    subG2 = dJ_dw(A2,X_2,Y2)
    subG3 = dJ_dw(A3,X_3,Y3)
    
    y_1 = (2/3) * y1 + (1/3) * y2 + (subG1 - subG10) 
    y_2 = (1/3) * y1 + (1/3) * y2 + (1/3) *y3 + (subG2 - subG20)
    y_3 = (1/3) * y2 + (2/3) * y3 + (subG3 - subG30)
    
    y1 = y_1
    y2 = y_2
    y3 = y_3
    
    X1 = X_1
    X2 = X_2
    X3 = X_3
    
    subG10 = subG1
    subG20 = subG2
    subG30 = subG3
    
    result1 = np.linalg.norm(X_opt-X1,2)
    result2 = np.linalg.norm(X_opt-X2,2)
    result3 = np.linalg.norm(X_opt-X3,2)
    
    arr1.append(result1)
    arr2.append(result2)
    arr3.append(result3)
    
    end = time.time()
    ttime += end-start
    
    r = (1/3) * (result1 + result2 + result3)
    
    if (i % 100 == 0):
        print('||%10d  |%17.10f     |%12.5f ||' %(i, r, ttime))


# In[ ]:





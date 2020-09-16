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

print("|| Iteration  |   primal residual    |     time    ||")
for i in range(1001):
    start = time.time()
    
    G = dJ_dw(A,X,Y)
    G1 = dJ_dw(A1,X1,Y1)
    G2 = dJ_dw(A2,X2,Y2)
    G3 = dJ_dw(A3,X3,Y3)


    X_1 = (2/3) * X1 + (1/3) * X2 - rho * G1
    X_2 = (1/3) * X1 + (1/3) * X2 + (1/3) *X3  - rho * G2
    X_3 = (1/3) * X2 + (2/3) * X3  - rho * G3

    X1 = X_1
    X2 = X_2
    X3 = X_3
    
    result1 = np.linalg.norm(X_opt-X1,2)
    result2 = np.linalg.norm(X_opt-X2,2)
    result3 = np.linalg.norm(X_opt-X3,2)
    
    end = time.time()
    ttime += end-start
    
    r = (1/3) * (result1 + result2 + result3)
    if (i % 100 == 0):
        print('||%10d  |%17.10f     |%12.5f ||' %(i, r, ttime))


# In[ ]:





# In[2]:


import matplotlib.pyplot as plt
plt.figure(dpi=100)
plt.subplot(411)
plt.grid()
plt.plot(X,'.-', color = 'black')
plt.subplot(412)
plt.plot(X1, '.-', color='red')
plt.grid()
plt.subplot(413)
plt.plot(X2, '.-' , color = 'blue')
plt.grid()
plt.subplot(414)
plt.plot(X3, '.-' , color = 'yellow')
plt.grid()
plt.show()


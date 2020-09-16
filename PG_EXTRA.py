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

n = 900
m = 1000

X= np.load('X_save.npy')
A= np.load('A_save.npy')

X1 = np.ones(m)
X2 = np.ones(m)
X3 = np.ones(m)
X_1old = X1
X_2old = X2
X_3old = X3

plt.figure(figsize=(25, 9), dpi=100)
plt.stem(X)

rho = 0.0001
Ndcs = 3
Y = A@X
lamb = 10


# In[ ]:





# In[2]:


get_ipython().run_line_magic('env', 'XLA_PYTHON_CLIENT_ALLOCATOR=platform')
gpus = jax.devices('gpu')

def cost1(X):
    Ap = A[0:300,:]
    Yp = Y[0:300]
    return (jnp.linalg.norm(jnp.dot(Ap,X) - Yp,2)**2) 

def cost2(X):
    Ap = A[300:600,:]
    Yp = Y[300:600]
    return (jnp.linalg.norm(jnp.dot(Ap,X) - Yp,2)**2) 

def cost3(X):
    Ap = A[600:n,:]
    Yp = Y[600:n]
    return (jnp.linalg.norm(jnp.dot(Ap,X) - Yp,2)**2) 
            
def Reg(X):
    return jnp.linalg.norm(lamb * X, 1)
            
G1 = grad(cost1)
L1 = jax.jit(cost1, device=gpus[0]) 
G1 = jax.jit(G1, device=gpus[0])

G2 = grad(cost2)
L2 = jax.jit(cost2, device=gpus[1])
G2 = jax.jit(G2, device=gpus[1])

G3 = grad(cost3)
L3 = jax.jit(cost3, device=gpus[2])
G3 = jax.jit(G3, device=gpus[2])


# In[3]:


import time

k=0
ttime = 0

start = time.time()
subG1 = G1(X1)
subG2 = G2(X2)
subG3 = G3(X3)

X12_1 = (2/3) * np.asarray(X1) + (1/3) * np.asarray(X2) - rho * np.asarray(subG1)
X12_2 = (1/3) * np.asarray(X1) + (1/3) * np.asarray(X2) + (1/3) * np.asarray(X3)  - rho * np.asarray(subG2)
X12_3 = (1/3) * np.asarray(X2) + (2/3) * np.asarray(X3) - rho * np.asarray(subG3)

X1_1 = np.maximum(X12_1-rho*lamb/3, np.zeros(m)) - np.maximum(-X12_1 -rho*lamb/3 , np.zeros(m)) 
X1_2 = np.maximum(X12_2-rho*lamb/3, np.zeros(m)) - np.maximum(-X12_2 -rho*lamb/3 , np.zeros(m)) 
X1_3 = np.maximum(X12_3-rho*lamb/3, np.zeros(m)) - np.maximum(-X12_3 -rho*lamb/3 , np.zeros(m))

end = time.time()

time_old = end - start


# In[4]:


print("|| Iteration  |       Loss      |     primal residual    |     time    ||")

for i in range(1,1001):   
    start = time.time()
    
    subG12 = G1(X1_1)
    subG22 = G2(X1_2)
    subG32 = G3(X1_3)

        
    X32_1 = (np.asarray(X12_1) + (2/3) * np.asarray(X1_1) + (1/3) * np.asarray(X1_2) -
             ((5/6) * np.asarray(X1) + (1/6) * np.asarray(X2)) - rho * (np.asarray(subG12)-np.asarray(subG1)))
    X32_2 = (np.asarray(X12_2) + (1/3) * np.asarray(X1_1) + (1/3) * np.asarray(X1_2) + (1/3) * np.asarray(X1_3) - ((1/6) * np.asarray(X1) + (4/6) * np.asarray(X2) + (1/6) * np.asarray(X3))  
            - rho *(np.asarray(subG22)-np.asarray(subG2)))
    X32_3 = (np.asarray(X12_3) + (1/3) * np.asarray(X1_2) + (2/3) * np.asarray(X1_3) - 
             ((1/6) * np.asarray(X2) + (5/6) * np.asarray(X3))- rho * (np.asarray(subG32)-np.asarray(subG3)))
    
    X2_1 = np.maximum(X32_1-rho*lamb/3, np.zeros(m)) - np.maximum(-X32_1 -rho*lamb/3 , np.zeros(m)) 
    X2_2 = np.maximum(X32_2-rho*lamb/3, np.zeros(m)) - np.maximum(-X32_2 -rho*lamb/3 , np.zeros(m))
    X2_3 = np.maximum(X32_3-rho*lamb/3, np.zeros(m)) - np.maximum(-X32_3 -rho*lamb/3 , np.zeros(m))
    
    X12_1 = X32_1
    X12_2 = X32_2
    x12_3 = X32_3
    
    X1 = X1_1
    X2 = X1_2
    X3 = X1_3
    
    X1_1 = X2_1
    X1_2 = X2_2
    X1_3 = X2_3
    
    subG1 = subG12
    subG2 = subG22
    subG3 = subG32
    
    result1 = np.linalg.norm(X-X1,2)
    result2 = np.linalg.norm(X-X2,2)
    result3 = np.linalg.norm(X-X3,2)
    
    r = (1/3) * (result1+result2+result3)

    end = time.time()
    ttime += end-start
    
    Loss = np.asarray(cost1(X1))+np.asarray(cost2(X2))+np.asarray(cost3(X3))+(1/3)*np.asarray(Reg(X1))+(1/3)*np.asarray(Reg(X2))+(1/3)*np.asarray(Reg(X3))
    if (i % 100 == 0):
        print('||%10d  |%14.1f   |  %17.10f     |%12.5f ||' %(i, Loss, r, time_old + ttime))


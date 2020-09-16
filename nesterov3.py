#!/usr/bin/env python
# coding: utf-8

# In[6]:


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


# In[7]:


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


# In[8]:


e=np.asarray(cost1(X))+np.asarray(cost2(X))+np.asarray(cost3(X))+(1/3)*np.asarray(Reg(X))+(1/3)*np.asarray(Reg(X))+(1/3)*np.asarray(Reg(X))


# In[9]:


import time

arr1 = []
arr2 = []
arr3 = []

k=0
ttime = 0

print("|| Iteration  |       Loss      |     primal residual    |     time    ||")
for i in range(3001):
    start = time.time()
    
    subG1 = G1(X1)
    subG2 = G2(X2)
    subG3 = G3(X3)
    
    q1 = X1 - rho*subG1
    q2 = X2 - rho*subG2
    q3 = X3 - rho*subG3
    
    q_1 = (2/3) * np.asarray(q1) + (1/3) * np.asarray(q2)
    q_2 = (1/3) * np.asarray(q1) + (1/3) * np.asarray(q2) + (1/3) * np.asarray(q3)
    q_3 = (1/3) * np.asarray(q2) + (2/3) * np.asarray(q3)
    
    X_1 = np.maximum(q_1-rho*lamb/Ndcs, np.zeros(m)) - np.maximum(-q_1 -rho*lamb/Ndcs, np.zeros(m)) 
    X_2 = np.maximum(q_2-rho*lamb/Ndcs, np.zeros(m)) - np.maximum(-q_2 -rho*lamb/Ndcs, np.zeros(m))
    X_3 = np.maximum(q_3-rho*lamb/Ndcs, np.zeros(m)) - np.maximum(-q_3 -rho*lamb/Ndcs, np.zeros(m))
    
    X1 = X_1 + i / (i+3) * (X_1 - X_1old)
    X2 = X_2 + i / (i+3) * (X_2 - X_2old)
    X3 = X_3 + i / (i+3) * (X_3 - X_3old)
    
    X_1old = X_1
    X_2old = X_2
    X_3old = X_3
    
    result1 = np.linalg.norm(X-X1,2)
    result2 = np.linalg.norm(X-X2,2)
    result3 = np.linalg.norm(X-X3,2)
    
    r = (1/Ndcs) * (result1+result2+result3)
    
    end = time.time()
    ttime += end-start
    
    Loss = np.asarray(cost1(X1))+np.asarray(cost2(X2))+np.asarray(cost3(X3))+(1/Ndcs)*np.asarray(Reg(X1))+(1/Ndcs)*np.asarray(Reg(X2))+(1/Ndcs)*np.asarray(Reg(X3))
    if (i % 100 == 0):
        print('||%10d  |%14.1f   |  %17.10f     |%12.5f ||' %(i, Loss, r, ttime))


# In[10]:


np.asarray(cost1(X))+np.asarray(cost2(X))+np.asarray(cost3(X))+(1/3)*np.asarray(Reg(X))+(1/3)*np.asarray(Reg(X))+(1/3)*np.asarray(Reg(X))


# In[ ]:





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

w1 = np.random.randn(m)
w2 = np.random.randn(m)
w3 = np.random.randn(m)

y1 = np.random.randn(m)
y2 = np.random.randn(m)
y3 = np.random.randn(m)

A1 = A[0:300,:]
A2 = A[300:700,:]
A3 = A[700:n,:]

Y = A@X
Y1 = Y[0:300]
Y2 = Y[300:700]
Y3 = Y[700:n]

X_opt = np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T), Y)
ttime = 0

# for i in range(250):
#     w1[i] = 1
#     w2[i] = 1
#     w3[i] = 1
    
# for i in range(250):
#     w1[250 + i] = -1
#     w2[250 + i] = -1
#     w3[250 + i] = -1

w1 = np.zeros(m)
w2 = np.zeros(m)
w3 = np.zeros(m)

def dJ_dw(A,X,Y):
    return 2*np.dot(A.T,(np.dot(A,X)-Y))

def f(x):
    return (s3*(x**3)) + (s2*(x**2)) + (s1*x) + s0


# In[2]:


L = [[1/3, -1/3, 0],[-1/3, 2/3, -1/3], [0, -1/3, 1/3]]

val, vec = np.linalg.eig(L)

sigma = val[1]

sigma


# In[3]:


# L = [[2/3, 1/3, 0],[1/3, 1/3, 1/3], [0, 1/3, 2/3]]

# val, vec = np.linalg.eig(L)

# sigma = val[1]

# sigma


# In[4]:


# from sympy import Symbol, solve 
# x=Symbol('x')
# equation = (x**3) + (0) * (x**2) + (0) * x + 1

# sol = np.array(solve(equation))

# sol.real
# # for i in range(2):
# #     if (2*sol[i]-(1-rho)*11) * (sol[i]-1+(rho**2)) < 0 and sol.imag[i] == 0:
# #         B = sol[i]

# # B


# In[5]:


from sympy import Symbol, solve 
x=Symbol('x')
rho = 0.5
etha = 1 + rho - 10 * (1-rho)

s0 = etha * ((1-rho**2)**2) * (etha-(3-etha)*etha*rho + 2*(1-etha)*(rho**2) + 2*(rho**3))
s1 = -1 * (1-(rho**2)) * ((etha**3)*rho + 4*(rho**5) - 2*etha*(rho**2)*(2*(rho**2)+rho-3) + (etha**2) * (4*(rho**3) - 4*(rho**2) - 6*rho + 3))
s2 = 3 * etha * ((1-rho)**2) * (1+rho) * (2*(rho**2) + etha)
s3 = (2*(rho**2) + etha) * (2*(rho**3)-etha)

print(s0)
print(s1)
print(s2)
print(s3)

a = np.roots([s3,s2,s1,s0])
a


# In[6]:


# rho = 0.96
# etha = 1 + rho - 10 * (1-rho)

# etha = 1 + rho -10*(1-rho)   
# s0 = etha * ((1-rho**2)**2) * (etha-(3-etha)*etha*rho + 2*(1-etha)*(rho**2) + 2*(rho**3))
# s1 = -1 * (1-(rho**2)) * ((etha**3)*rho + 4*(rho**5) - 2*etha*(rho**2)*(2*(rho**2)+rho-3) + (etha**2) * (4*(rho**3) - 4*(rho**2) - 6*rho + 3))
# s2 = 3 * etha * ((1-rho)**2) * (1+rho) * (2*(rho**2) + etha)
# s3 = (2*(rho**2) + etha) * (2*(rho**3)-etha)
    
# root = np.roots([s3,s2,s1,s0])
# root


# In[7]:


rho1 = 0
rho2 = 1
rho = 0.5

sigma = 0.818
sigma_hat = sigma

while (rho2 - rho1) > 0.0001:
    rho = (rho1+rho2) / 2 
      
    etha = 1 + rho -10*(1-rho)   
    s0 = etha * ((1-rho**2)**2) * (etha-(3-etha)*etha*rho + 2*(1-etha)*(rho**2) + 2*(rho**3))
    s1 = -1 * (1-(rho**2)) * ((etha**3)*rho + 4*(rho**5) - 2*etha*(rho**2)*(2*(rho**2)+rho-3) + (etha**2) * (4*(rho**3) - 4*(rho**2) - 6*rho + 3))
    s2 = 3 * etha * ((1-rho)**2) * (1+rho) * (2*(rho**2) + etha)
    s3 = (2*(rho**2) + etha) * (2*(rho**3)-etha)
    
    root = np.roots([s3,s2,s1,s0])
    
    for i in range(3):
        if (2*root[i]-(1-rho)*11) * (root[i]-1+rho**2) < 0 and (root[i].imag == 0):
            beta = root[i]
    
    sigma_Hat = (rho**2) * ((beta-1+(rho**2)) / (beta-1+rho)) * ((2*(rho**2) + etha) * beta - (1-rho**2) * etha) / ((1+rho) * (etha-2*etha*rho+2*(rho**2))-(2*(rho**2)+etha)*beta)
    sigma_Hat = sigma_Hat ** 0.5
    
    print(sigma_Hat)
    if (sigma_Hat < sigma) :
        rho1 = rho
    else:
        rho2 = rho
        
rho = np.maximum(rho,0.818)


# In[8]:


import time

print(rho, beta)
print("|| Iteration  |   primal residual    |     time    ||")

alpha = (1 - rho) / 100
gamma = (beta + 1) 

for i in range(1001):  
    start = time.time()
    
    v1 = (1/3) * X1 + (-1/3) * X2
    v2 = (-1/3) * X1 + (2/3) * X2 + (-1/3) *X3
    v3 = (-1/3) * X2 + (1/3) * X3
    
    y1 = X1 - v1
    y2 = X2 - v2
    y3 = X3 - v3
    
    u1 = dJ_dw(A1,y1,Y1)
    u2 = dJ_dw(A2,y2,Y2)
    u3 = dJ_dw(A3,y3,Y3)
    
    X1_2 = X1 + beta * w1 - alpha * u1 - gamma * v1
    X2_2 = X2 + beta * w2 - alpha * u2 - gamma * v2
    X3_2 = X3 + beta * w3 - alpha * u3 - gamma * v3
    
    w1_2 = w1 - v1
    w2_2 = w2 - v2
    w3_2 = w3 - v3
    
    w1 = w1_2
    w2 = w2_2
    w3 = w3_2
    
    X1 = X1_2
    X2 = X2_2
    X3 = X3_2
    
    result1 = np.linalg.norm(X_opt-X1,2)
    result2 = np.linalg.norm(X_opt-X2,2)
    result3 = np.linalg.norm(X_opt-X3,2)
    
    end = time.time()
    ttime += end-start
    
    r = (1/3) * (result1 + result2 + result3)
    if (i % 100 == 0):
        print('||%10d  |%17.10f     |%12.5f ||' %(i, r, ttime))


# In[ ]:





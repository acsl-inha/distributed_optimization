#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cvxpy as cp
import random
import time


# In[10]:


etha = 0.001 # etha(step size)
sigma = 0.2 # spertral Gap

A1 = np.array([[1,-0.5,etha],[1,0,0],[0,0,0]]) # matrix A1
B1 = np.array([[-etha,1,1],[0,0,0],[1,0,0]]) # matrix B1

A2 = np.array([[2,-1,etha], [1,0,0], [0,0,0]]) # matrix A2
B2 = np.array([[-etha,1,1],[0,0,0],[1,0,0]]) # matrix B2

CD_0e_11 = np.array([[1,0,0,0,0,0],[0,0,0,1,0,0]]) # [[C1,D1], [0,e1]]  i=1, j=1
CD_0e_21 = np.array([[1,0,0,0,0,0],[0,0,0,0,1,0]]) # [[C1,D1], [0,e2]]  i=1, j=2
CD_0e_31 = np.array([[0,-0.5,0,0,0,0], [0,0,0,0,0,1]]) # [[C1,D1], [0,e3]] i=1, j=3

CD_0e_12 = np.array([[1,0,0,0,0,0],[0,0,0,1,0,0]]) # [[C2,D2], [0,e1]] i=2, j=1
CD_0e_22 = np.array([[1,0,0,0,0,0],[0,0,0,0,1,0]]) # [[C2,D2], [0,e2]] i=2, j=2
CD_0e_32 = np.array([[0,-0.5,0,0,0,0], [0,0,0,0,0,1]]) # [[C2,D2], [0,e3]] i=2, j=3

L = 1000
m = 100

M1 = np.array([[-2*m*L, m+L],[m+L,-2]]) # matrix M1 or marix M2   j = 1 
Mj = np.array([[sigma**2,0],[0,-1]]) # matrix M1 or marix M2   j = 2,3

R1 = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]) # basis for null[F1 G1] 
R2 = np.array([[1,-etha,0],[1,0,0],[0,1,0],[0,0,1],[0,0,0],[0,0,0]]) # basis for null[F2 G2]


# In[14]:


rho = cp.Variable(1)
lamb = cp.Variable(3)

n = 3
P1 = cp.Variable((n,n), symmetric=True)
P2 = cp.Variable((n,n), symmetric=True)

constraints = ([R1.T@(cp.vstack([cp.hstack([A1.T@P1@A1-(rho**2)*P1, A1.T@P1@B1]),cp.hstack([B1.T@P1@A1, B1.T@P1@B1])]) 
                + lamb[0] * CD_0e_11.T@M1@CD_0e_11 + lamb[1] * CD_0e_21.T@Mj@CD_0e_21 + lamb[2] * CD_0e_31.T@Mj@CD_0e_31)@R1 << 0])

constraints += ([R2.T@(cp.vstack([cp.hstack([A2.T@P2@A2-(rho**2)*P2, A2.T@P2@B2]),cp.hstack([B2.T@P2@A2, B2.T@P2@B2])]) 
                + lamb[0] * CD_0e_12.T@M1@CD_0e_12 + lamb[1] * CD_0e_22.T@Mj@CD_0e_22 + lamb[2] * CD_0e_32.T@Mj@CD_0e_32)@R2 << 0])

constraints += [rho >= 0]
constraints += [lamb >= 0]
constraints += [P1 >> 0]
constraints += [P2 >> 0]

prob = cp.Problem(cp.Minimize(1), constraints)
prob.solve()

print("The optimal value is", prob.value)
print("A solution P is")
print(P1.value)


# In[ ]:


for i in range(1000):
    rho = i/1000
    
    lamb = cp.Variable(3)

    n = 3
    P1 = cp.Variable((n,n), symmetric=True)
    P2 = cp.Variable((n,n), symmetric=True)

    constraints = ([R1.T@(cp.vstack([cp.hstack([A1.T@P1@A1-(rho**2)*P1, A1.T@P1@B1]),cp.hstack([B1.T@P1@A1, B1.T@P1@B1])]) 
                    + lamb[0] * CD_0e_11.T@M1@CD_0e_11 + lamb[1] * CD_0e_21.T@Mj@CD_0e_21 + lamb[2] * CD_0e_31.T@Mj@CD_0e_31)@R1 << 0])

    constraints += ([R2.T@(cp.vstack([cp.hstack([A2.T@P2@A2-(rho**2)*P2, A2.T@P2@B2]),cp.hstack([B2.T@P2@A2, B2.T@P2@B2])]) 
                    + lamb[0] * CD_0e_12.T@M1@CD_0e_12 + lamb[1] * CD_0e_22.T@Mj@CD_0e_22 + lamb[2] * CD_0e_32.T@Mj@CD_0e_32)@R2 << 0])

    constraints += [lamb >= 0]
    constraints += [P1 >> 0]
    constraints += [P2 >> 0]

    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve()
    
    print("A solution P is")
    print(P1.value)


# In[ ]:





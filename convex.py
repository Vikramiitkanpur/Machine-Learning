#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import math


# In[10]:


x_seen = np.load('X_seen.npy', allow_pickle=True, encoding='bytes')
attr_s = np.load('class_attributes_seen.npy', allow_pickle=True, encoding='bytes')
attr_u = np.load('class_attributes_unseen.npy', allow_pickle=True, encoding='bytes')
x_test = np.load('Xtest.npy', allow_pickle=True, encoding='bytes')
y_test = np.load('Ytest.npy', allow_pickle=True, encoding='bytes')


# In[11]:


# Calculating Mu for the 40 seen classes
mean_k = [] 
for i in range(40):
    mean_k.append(list(np.average(x_seen[i], axis=0)))


# In[12]:


# Calculating similarity matrix (similarity vector of each unseen class as rows)
s = []
for i in range(10):
    b_c = list(attr_u[i])
    b_c = []
    for j in range(40):
        a_k = list(attr_s[j])
        b_ck = sum([a_k[d]*b_c[d] for d in range(85)])  # Inner product of b_c and a_k(similarity of class c and class k)
        b_c.append(b_ck)
    s_sum = sum(b_c)
    b_c = [t/s_sum for t in b_c]  # Normalizing the values (sum = 1)
    s.append(b_c)


# In[13]:


# Calculating the mu_c for unseen classes 
mu_c = []
for i in range(10):
    s_c = s[i]
    temp = []
    for j in range(40):
        smu = [s_c[j]*m for m in mu_k[j]]
        temp.append(smu)
    temp = np.sum(temp, axis=0)
    mu_c.append(temp)


# In[14]:


# Function to find l2 distance between two points
def l2_dist(a, b):
    ans = 0
    for i in range(4096):
        ans += (a[i]-b[i])*(a[i]-b[i])
    return math.sqrt(ans)


# In[15]:


def pred(a):
    distances = []
    for m in mu_c:
        d = l2_dist(a, m)
        distances.append(d)
    # print(distances)
    return np.argmin(distances)+1


# In[16]:


predictions = []
count = 0
for t in x_test:
    predictions.append(pred(t))
    count += 1


# In[17]:


correct = 0
for i in range(6180):
    if predictions[i] == y_test[i]:
        #print(predictions[i])
        correct+=1
        

acc=100*correct/6180
print(acc)


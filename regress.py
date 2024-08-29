#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import math


# In[ ]:


x_seen = np.load('X_seen.npy', allow_pickle=True, encoding='bytes')
attr_s = np.load('class_attributes_seen.npy', allow_pickle=True, encoding='bytes')
attr_u = np.load('class_attributes_unseen.npy', allow_pickle=True, encoding='bytes')
x_test = np.load('Xtest.npy', allow_pickle=True, encoding='bytes')
y_test = np.load('Ytest.npy', allow_pickle=True, encoding='bytes')


# In[ ]:


mu_k = [] 
for i in range(40):
    mu_k.append(list(np.average(x_seen[i], axis=0)))
mu_k = np.array(mu_k)


# In[ ]:


# Function to find l2 distance between two points
def l2_dist(a, b):
    ans = 0
    for i in range(4096):
        ans += (a[i]-b[i])*(a[i]-b[i])
    return math.sqrt(ans)


# In[ ]:


def pred(a, mu_c):
    distances = []
    for m in mu_c:
        d = l2_dist(a, m)
        distances.append(d)
    return np.argmin(distances)+1


# In[ ]:


# Calculate the w for all lambda values

mu = []

for lambda_ in [0.01, 0.1, 1, 10, 20, 50, 100]:
    t1 = np.matmul(np.transpose(attr_s), attr_s) 
    t2 = np.add(t1, lambda_*np.identity(85))
    t3 = np.linalg.inv(t2)
    t4 = np.matmul(t3, np.transpose(attr_s))

    W = np.matmul(t4, mu_k)

    mu.append(np.matmul(attr_u, W))


# In[ ]:


pred_list = []
for m in mu:
    predictions = []
    count = 0
    for t in x_test:
        predictions.append(pred(t,m))
        count += 1
    pred_list.append(predictions)


# In[ ]:


l = [0.01, 0.1, 1, 10, 20, 50, 100]
j = 0
for predictions in pred_list:   
    correct = 0
    for i in range(6180):
        if predictions[i] == y_test[i]:
            correct+=1
    print("lamda =", l[j], ":", correct)
    j+=1


# In[ ]:





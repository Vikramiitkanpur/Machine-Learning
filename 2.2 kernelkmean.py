#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

def predict_cluster(m, n):
    distances = np.zeros((m.shape[0], n.shape[0]))
    for i in range(n.shape[0]):
        diff = m - n[i, :].reshape((1, -1))
        distances[:, i] = np.sum(np.square(diff), axis=1)
    cluster_labels = np.argmin(distances, axis=1)
    return cluster_labels.reshape(-1, 1)

def compute_cluster_means(x, cluster_labels):
    cluster_means = np.zeros((2, x.shape[1]))
    cluster_means[0, :] = np.mean(x[cluster_labels == 0], axis=0)
    cluster_means[1, :] = np.mean(x[cluster_labels == 1], axis=0)
    return cluster_means

x_data = np.genfromtxt('kmeans_data.txt', delimiter='  ')

for iteration in range(10):
    random_index = np.random.randint(250, size=1).reshape(())
    feature_vector = np.exp(-0.1 * np.sum(np.square(x_data - x_data[random_index, :].reshape((1, -1))), axis=1)).reshape(-1, 1)
    
    cluster_centers = feature_vector[:2, :]
    cluster_labels = predict_cluster(feature_vector, cluster_centers)
    
    cluster_centers = compute_cluster_means(feature_vector, cluster_labels)
    cluster_labels = predict_cluster(feature_vector, cluster_centers)
    
    positive_samples = (cluster_labels == 1).reshape(cluster_labels.shape[0])
    negative_samples = (cluster_labels == 0).reshape(cluster_labels.shape[0])

    plt.figure(iteration)
    plt.scatter(x_data[positive_samples, 0], x_data[positive_samples, 1], c='r')
    plt.scatter(x_data[negative_samples, 0], x_data[negative_samples, 1], c='g')
    plt.plot(x_data[random_index, 0], x_data[random_index, 1], 'r*')
    
    # Save the figure to a file
    plt.savefig(f'iteration_{iteration}.png')

# Save cluster centers to text file
np.savetxt('cluster_centers.txt', cluster_centers)

plt.show()


# In[ ]:





# In[ ]:





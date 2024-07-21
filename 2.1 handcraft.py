#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def calculate_distances(x, u):
    distances = np.zeros((x.shape[0], u.shape[0]))
    for i in range(u.shape[0]):
        diff = x - u[i, :].reshape((1, -1))
        distances[:, i] = np.sum(np.square(diff), axis=1)
    return distances


# In[3]:


def assign_clusters(distances):
    return np.argmin(distances, axis=1).reshape(-1, 1)


# In[4]:


def update_means(x, c):
    means = np.zeros((2, x.shape[1]))
    for i in range(2):
        means[i, :] = np.mean(x[c == i], axis=0)
    return means


# In[5]:


def plot_clusters(x, c):
    positive_points = x[np.squeeze(c == 1)]
    negative_points = x[np.squeeze(c == 0)]

    plt.scatter(positive_points[:, 0], positive_points[:, 1], c='r')
    plt.scatter(negative_points[:, 0], negative_points[:, 1], c='g')
    plt.savefig('kmeans_clusters.png')
    plt.show()


# In[6]:


def main():
    # Load data
    x = np.genfromtxt('kmeans_data.txt', delimiter='  ')

    # Feature extraction
    fx = (np.sum(np.square(x), axis=1)).reshape(-1, 1)

    # Initialize centroids
    centroids = fx[:2, :]

    # Initial cluster assignment
    clusters = assign_clusters(calculate_distances(fx, centroids))

    # K-means iterations
    for iteration in range(10):
        # Update centroids
        centroids = update_means(fx, clusters)

        # Reassign clusters
        clusters = assign_clusters(calculate_distances(fx, centroids))

    # Plot final clusters
    plot_clusters(x, clusters)
    
    # Save centroids and clusters to text files
    np.savetxt('centroids.txt', centroids)
    np.savetxt('clusters.txt', clusters)

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





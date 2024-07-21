#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

with open('mnist_small.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']  # Features
Y = data['Y']  # Labels


# In[4]:


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plotting PCA
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y, cmap='viridis')
plt.title('PCA Projection')

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
plt.show()
# Plotting t-SNE
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y, cmap='viridis')
plt.title('t-SNE Projection')
plt.show()


# In[ ]:







# In[ ]:





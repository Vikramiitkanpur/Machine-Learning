#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

def load_data(data_location):
    # Loads the data from the specified location
    # Usage: x_train, y_train, x_test, y_test = load_data('data')
    train_data = np.genfromtxt('ridgetrain.txt', delimiter='  ')
    test_data = np.genfromtxt('ridgetest.txt', delimiter='  ')
    return train_data[:, 0], train_data[:, 1], test_data[:, 0], test_data[:, 1]

def compute_kernel(x, y):
    # Computes the kernel function
    # Usage: K = compute_kernel(x_train, x_train)
    return np.exp(-0.1 * np.square(x.reshape((-1, 1)) - y.reshape((1, -1))))

x_train, y_train, x_test, y_test = load_data('data')
kernel_matrix = compute_kernel(x_train, x_train)
iteration_values = [ 0.1, 1, 10, 100]
identity_matrix = np.eye(x_train.shape[0])

for regularization_param in iteration_values:
     # Compute the kernel for the test set
    kernel_test = compute_kernel(x_train, x_test)
    
    # Regularization using the kernel matrix and identity matrix
    regularization_matrix = np.dot(np.linalg.inv(kernel_matrix + regularization_param * identity_matrix), y_train.reshape((-1, 1)))
    
   
    
    # Make predictions using the regularization matrix and test kernel
    y_prediction = (np.dot(regularization_matrix.T, kernel_test)).reshape((-1, 1))

    # Compute root mean squared error
    rmse = np.sqrt(np.mean(np.square(y_test.reshape((-1, 1)) - y_prediction)))
    
    # Display results
    print('RMSE for regularization parameter = ' + str(regularization_param) + ' is ' + str(rmse))

    # Plotting results
    plt.figure(regularization_param)
    plt.title('Regularization parameter = ' + str(regularization_param) + ', RMSE = ' + str(rmse))
    plt.plot(x_test, y_prediction, 'b*', label='Predicted')
    plt.plot(x_test, y_test, 'r*', label='True')
    plt.legend()

plt.show()


# In[ ]:





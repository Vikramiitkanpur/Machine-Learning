#!/usr/bin/env python
# coding: utf-8

# In[4]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

def load_data(directory):
    # Load data by specifying the data folder location
    # Example usage: x_train, y_train, x_test, y_test = load_data('dataset')
    train_data = np.genfromtxt('ridgetrain.txt', delimiter='  ')
    test_data = np.genfromtxt('ridgetest.txt', delimiter='  ')
    return train_data[:, 0], train_data[:, 1], test_data[:, 0], test_data[:, 1]

def compute_landmark_matrix(x, y):
    # Compute the landmark matrix: landmark_matrix(x_train, x_train)
    return np.exp(-0.1 * np.square(x.reshape((-1, 1)) - y.reshape((1, -1))))

x_train, y_train, x_test, y_test = load_data('data')

lambda_values = [2, 5, 20, 50, 100]

for lambda_val in lambda_values:
    # Randomly select a subset of landmarks
    landmark_subset = np.random.choice(x_train, lambda_val, replace=False)
    identity_matrix = np.eye(lambda_val)
    
    # Compute the landmark matrix for the training set
    landmark_matrix_train = compute_landmark_matrix(x_train, landmark_subset)

    # Calculate weights using the selected landmarks
    weights = np.dot(
        np.linalg.inv(np.dot(landmark_matrix_train.T, landmark_matrix_train) + 0.1 * identity_matrix),
        np.dot(landmark_matrix_train.T, y_train.reshape((-1, 1)))
    )

    # Compute the landmark matrix for the test set
    landmark_matrix_test = compute_landmark_matrix(x_test, landmark_subset)

    # Make predictions using the weights and test landmarks
    y_predicted = np.dot(landmark_matrix_test, weights)

    # Compute root mean squared error
    root_mean_squared_error = np.sqrt(np.mean(np.square(y_test.reshape((-1, 1)) - y_predicted)))
    print('RMSE for lambda = ' + str(lambda_val) + ' is ' + str(root_mean_squared_error))

    # Plotting results
    plt.figure(lambda_val)
    plt.title('Lambda = ' + str(lambda_val) + ', RMSE = ' + str(root_mean_squared_error))
    plt.plot(x_test, y_predicted, 'r*', label='Predicted')
    plt.plot(x_test, y_test, 'b*', label='True')
    plt.legend()

    # Save the figure to a file
    plt.savefig(f'figure_lambda_{lambda_val}.png')

    # Save the predicted values to a text file
    np.savetxt(f'predicted_values_lambda_{lambda_val}.txt', y_predicted)

# Save data to text files
np.savetxt('x_test.txt', x_test)
np.savetxt('y_test.txt', y_test)

plt.show()


# In[ ]:






# In[ ]:





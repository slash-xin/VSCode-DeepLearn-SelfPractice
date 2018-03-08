# -*- encoding: utf-8 -*-
# Date: 8 Mar 2018
# Author: Slash.Xin
# Descprition: Optimization Method


import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


#-------------------------
# 1. Gradient Descent
#-------------------------
def update_parameters_with_gd(parameters, grads, learning_rate):
    '''
    Update parameters using one step of gradient descent

    Arguments:
    arameters -- python dictionary containing your parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients to update each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    learning_rate -- the learning rate, scalar.
    
    Returns:
    parameters -- python dictionary containing your updated parameters
    '''
    L = len(parameters) // 2

    for l in range(L):
        parameters['W'+str(l+1)] -= learning_rate * grads['dW'+str(l+1)]
        parameters['b'+str(l+1)] -= learning_rate * grads['db'+str(l+1)]
    
    return parameters

# Test the function
parameters, grads, learning_rate = update_parameters_with_gd_test_case()
parameters = update_parameters_with_gd(parameters, grads, learning_rate)
print('W1 =', parameters['W1'])
print('b1 =', parameters['b1'])
print('W2 =', parameters['W2'])
print('b2 =', parameters['b2'])

#----------------------------------
# 2. Mini-Batch Gradient Descent
#----------------------------------
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    '''
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    '''

    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: Partition
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_x = shuffled_X[:, k * mini_batch_size: (k+1) * mini_batch_size]
        mini_batch_y = shuffled_Y[:, k * mini_batch_size: (k+1) * mini_batch_size]

        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_x = shuffled_X[:, mini_batch_size * num_complete_minibatches: m]
        mini_batch_y = shuffled_Y[:, mini_batch_size * num_complete_minibatches: m]

        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)
    
    return mini_batches

# Test the function
X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)
print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape)) 
print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))
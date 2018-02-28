# -*- encoding: utf-8 -*-
# Date: 28 Feb 2018
# Author: Slash.Xin
# Descprition: Deep Hidden Layer Neural Network



# ----------------------
# 1. Package Import
# ----------------------
import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v2 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

# Magic Function
#%matplotlib inline

# set default configuration of plots
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Magic Function
#%load_ext autoreload
#%autoreload 2

# Set the random seed
np.random.seed(1)



# --------------------------------------------------------------------------------------------
# 2. Initialization: two helper functions that will initialize the parameters for the model.
#    The first function will be used to initialize parameters for a two layer model.
#    The second function will be used to initialize parameters for a L layer model.
# --------------------------------------------------------------------------------------------
# 2.1. 2-layer Neural Network
def initialize_parameters(n_x, n_h, n_y):
    '''
    Arguments:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the outpu layer

    Returns:
    paramters -- python dictionary containing the parameters:
                   W1 -- weight matrix of shape (n_h, n_x)
                   b1 -- bais vector of shape (n_h, 1)
                   W2 -- wight matrix of shape (n_y, n_h)
                   b2 -- bais vector of shape (n_y, 1)
    '''
    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y, 1))

    parameters = {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2}

    return parameters

# Test function
parameters = initialize_parameters(2, 2, 1)
print('W1 =', parameters['W1'])
print('b1 =', parameters['b1'])
print('W2 =', parameters['W2'])
print('b2 =', parameters['b2'])


# --------------------------------
# 2.2. L-layer Neural Network
# --------------------------------
def initialize_parameters_deep(layer_dims):
    '''
    Arguments:
    layer_dims -- python array (list) containing the dimension of each layer in model.

    Returns:
    parameters -- python dictionary containing the parameters 'W1', 'b1', 'W2', 'b2', .... 'WL', 'bL';
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    '''

    np.random.seed(3)
    parameters = {}

    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    return parameters

# Test the funcion
parameters = initialize_parameters_deep([5, 4, 3])
print('W1 =', parameters['W1'])
print('b1 =', parameters['b1'])
print('W2 =', parameters['W2'])
print('b2 =', parameters['b2'])




# --------------------------------------------------------------------------------------------
# 3. Forward Propagation Module:
#    First, implementing some basic functions which will be used when implementing the model.
#      (1) LINEAR
#      (2) LINEAR -> ACTIVATION where ACTIVATION will be wither Relu or Sigmoid
#      (3) [LINEAR -> RELU] X(L-1) -> LINEAR -> SIGMOID (whole model)
# --------------------------------------------------------------------------------------------
# 3.1. Linear Forward

def linear_forward(A, W, b):
    '''
    Implement the linear part of layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bais vector, numpu array of shape (size of current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing 'A', 'W' and 'b'; stored for computing the backward pass efficiently
    '''
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache

# Test the function
A, W, b = linear_forward_test_case()
Z, linear_cache = linear_forward(A, W, b)
print('Z =', Z)
print('linear_cache =', linear_cache)


# --------------------------------
# 3.1. Linear-Activation Forward
# --------------------------------
def linear_activation_forward(A_prev, W, b, activation):
    '''
    Implement the forward propgation for the LINEAR -> ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: 'sigmoid' or 'relu'

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing 'linear_cache' and 'activation_cache';
             stored for computing the backward pass effienciently
    '''

    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)
    return A, cache

# Test the function
A_prev, W, b = linear_activation_forward_test_case()

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, 'sigmoid')
print('with sigmoid: A=', A)
A, linear_activation_cache = linear_activation_forward(A_prev, W, b, 'relu')
print('with relu: A=', A)
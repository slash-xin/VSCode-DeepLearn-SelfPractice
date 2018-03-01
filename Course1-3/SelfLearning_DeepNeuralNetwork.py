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



# L-Layer Model: write a function that replicates the previous one (linear_activation_forward with RELU)
# L-1 time.
def L_model_forward(X, parameters):
    '''
    Implement forward propgation for the [LINEAR-RELU]*(L-1)->LINER->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of example)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                 every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                 the cache of linear_sigmoid_forward() (there is one, index L-1)
    '''

    caches = []
    A = X
    L = len(parameters) // 2 # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add 'cache' to the 'caches' list
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], 'relu')
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add 'cache' to the 'caches' list.
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], 'sigmoid')
    caches.append(cache)

    return AL, caches

# Test the function
X, parameters = L_model_forward_test_case()
AL, caches = L_model_forward(X, parameters)
print('AL =', AL)
print('caches =', caches)




# --------------------------------
# 4. Cost Function
# --------------------------------
def compute_cost(AL, Y):
    '''
    Implement the cost function defined by equation.
    Arguments:
    AL -- probabilities vector corresponding to the label predictions, shape of (1, number of examples.)
    Y -- the true 'label' vector (for example, containing 0 if non-cat; 1 if cat), shape of (1, number of example)
    Returns:
    cost -- cross-entropy cost
    '''
    m = Y.shape[1]

    cost = -1.0 / m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL)), axis=1, keepdims=True)
    cost = np.squeeze(cost)

    return cost

# Test the function
Y, AL = compute_cost_test_case()
cost = compute_cost(AL, Y)
print('cost =', cost)



# -----------------------------------------------------------------------------------------------------------
# 5. Backward Propagation Module
#    First, implement some helper functions for backward propagation just like the forward propagation.
#      (1) LINEAR backward
#      (2) LINEAR -> ACTIVATION backward where ACTIVATION computes the derivative of either the Relu or
#          sigmoid activation
#      (3) [LINEAR -> RELU]X(L-1) -> LINEAR -> SIGMOID backward (whole model)
# -----------------------------------------------------------------------------------------------------------
# 5.1. Linear backward
def linear_backward(dZ, cache):
    '''
    Implement the linear protion of backward propagation for a single layer (layer l)
    
    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    '''
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1.0 / m * np.dot(dZ, A_prev.T)
    db = 1.0 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

# Test the function
dZ, linear_cache = linear_backward_test_case()
dA_prev, dW, db = linear_backward(dZ, linear_cache)
print('dA_prev =', dA_prev)
print('dW =', dW)
print('db =', db)

# ---------------------------------
# 5.2. Linear-Activation backward
# ---------------------------------
def linear_activation_backward(dA, cache, activation):
    '''
    Implement the backward propagation for LINEAR -> ACTIVATION layer.
    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: 'relu' or 'sigmoid'

    Returns:
    dA_prev -- Gradient fo the cost with respect to the activation (of the previous layer l-1), same as shape A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    '''
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

# Test the function
AL, linear_activation_cache = linear_activation_backward_test_case()

dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, 'sigmoid')
print('sigmoid:')
print('dA_prev =', dA_prev)
print('dW =', dW)
print('db =', db)

dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, 'relu')
print('relu:')
print('dA_prev =', dA_prev)
print('dW =', dW)
print('db =', db)


# ---------------------------------
# 5.3. L-Model Backward
# ---------------------------------
def L_model_backward(AL, Y, caches):
    '''
    Implement the backward propagation for the [LINEAR -> RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_Forward())
    Y -- true 'label' vector (containing 0 if non-cat; 1 if cat)
    caches -- list of caches containing:
                  every cache of linear_activation_forward() with 'relu' (it's caches[l], for l in range(L-1), i.e l = 0 ... L-2)
                  the cache of linear_activation_forward() with 'sigmoid' (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads['dA'+str(l)] = ...
             grads['dW'+str(l)] = ...
             grads['db'+str(l)] = ...
    '''
    grads = {}
    L = len(caches) # number of layers
    #m = AL.shape[1] # number of examples
    Y = Y.reshape(AL.shape) # after this line, Y is same shape as AL

    dAL = -(np.divide(Y, AL)) - np.divide(1-Y, 1-AL)

    current_cache = caches[L-1]
    grads['dA'+str(L)], grads['dW'+str(L)], grads['db'+str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA'+str(l+2)], current_cache, 'relu')
        grads['dA'+str(l+1)] = dA_prev_temp
        grads['dW'+str(l+1)] = dW_temp
        grads['db'+str(l+1)] = db_temp
    
    return grads

# Test the function
AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)
print ("dW1 = ", grads["dW1"])
print ("db1 = ", grads["db1"])
print ("dA1 = ", grads["dA1"])



# ---------------------------------
# 5.4. Update Parameters
# ---------------------------------
def update_parameters(parameters, grads, learning_rate):
    '''
    Update parameters using gradient descent

    Argument:
    parameter -- python dictionary containing the parameters
    grads -- python dictionary containing the gradient, output of L_model_backward()

    Returns:
    parameters -- python dictionary containing the updated paremters
                    parameters['W'+str(l)] = ...
                    parameters['b'+str(l)] = ...
    '''

    L = len(parameters) // 2 #number of layers in the neural network

    for l in range(L):
        parameters['W'+str(l+1)] -= learning_rate * grads['dW'+str(l+1)]
        parameters['b'+str(l+1)] -= learning_rate * grads['db'+str(l+1)]
    
    return parameters

# Test the function
parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)
print('W1 =', parameters['W1'])
print('b1 =', parameters['b1'])
print('W2 =', parameters['W2'])
print('b2 =', parameters['b2'])
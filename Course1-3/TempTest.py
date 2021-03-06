# -*- encoding: utf-8 -*-
# Date: 28 Feb 2018
# Author: Slash.Xin
# Descprition: Deep Hidden Layer Neural Network



# ----------------------
# 1. Package Import
# ----------------------
import numpy as np
from testCases_v2 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward


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
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2}

    return parameters

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

    np.random.seed(1)
    parameters = {}

    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    return parameters



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

    #print('===================== shape of A_prev:', A_prev.shape)

    dW = 1.0 / m * np.dot(dZ, A_prev.T)
    db = 1.0 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

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
    
    #print('===================== shape of dZ:', dZ.shape)
    #print('===================== shape of dW:', dW.shape)
    #print('===================== shape of db:', db.shape)
    #print('===================== shape of dA_prev:', dA_prev.shape)
    
    return dA_prev, dW, db


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

    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))

    #print('=====================BackwardPropagation: AL:', AL[0][:5])
    #print('=====================BackwardPropagation: Y:', Y[0][:5])
    #print('=====================BackwardPropagation: dAL:', dAL[0][:5])
    current_cache = caches[L-1]
    #print('==================Current Cache============', current_cache[0][0][:5])
    grads['dA'+str(L)], grads['dW'+str(L)], grads['db'+str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')

    #print('=====================BackwardPropagation: dA{0}, shape:{1}'.format(L, grads['dA'+str(L)].shape))
    #print('=====================BackwardPropagation: dW{0}, shape:{1}'.format(L, grads['dW'+str(L)].shape))
    #print('=====================BackwardPropagation: db{0}, shape:{1}'.format(L, grads['db'+str(L)].shape))

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA'+str(l+2)], current_cache, 'relu')
        grads['dA'+str(l+1)] = dA_prev_temp
        grads['dW'+str(l+1)] = dW_temp
        grads['db'+str(l+1)] = db_temp
    
    return grads
def L_model_backward2(AL, Y, caches):
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

    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    grads['dA' + str(L)] = dAL

    #print('=====================BackwardPropagation: dAL, shape:', dAL.shape)
    current_cache = caches[L-1]
    #print('==================Current Cache============', len(current_cache))
    grads['dA'+str(L-1)], grads['dW'+str(L)], grads['db'+str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')

    #print('=====================BackwardPropagation: dA{0}, shape:{1}'.format(L, grads['dA'+str(L)].shape))
    #print('=====================BackwardPropagation: dW{0}, shape:{1}'.format(L, grads['dW'+str(L)].shape))
    #print('=====================BackwardPropagation: db{0}, shape:{1}'.format(L, grads['db'+str(L)].shape))

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA'+str(l+1)], current_cache, 'relu')
        grads['dA'+str(l)] = dA_prev_temp
        grads['dW'+str(l+1)] = dW_temp
        grads['db'+str(l+1)] = db_temp
    
    return grads


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



# ---------------------------------
# 6. Use Model to Predict 
# ---------------------------------
def predict(X, Y, parameters):
    '''
    This function is used to predict the results of a L-layer neural network.

    Arguments:
    X -- data set of examples will be predicted
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    '''
    m = X.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1, m))

    probas, caches = L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    
    print('Accuracy:', np.sum((p == Y)/m))

    return p













# -------------------------------------------------------------------
# Deep Neural Network for Image Classification: Application
# ---------------------------------------- --------------------------
# Import the packages
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage



# Load Dataset
def load_data(path):
    train_dataset = h5py.File('{0}/train_catvnoncat.h5'.format(path), "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('{0}/test_catvnoncat.h5'.format(path), "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

train_x_orig, train_y, test_x_orig, test_y, classes =load_data('course1-3/datasets')


# Reshape the training and test data
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

print('shape of train_x:', train_x.shape)
print('shape of test_x:', test_x.shape)


# two layer model
def two_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    '''
    Implement a two-layer neural network: LINEAR -> RELU -> LINEAR -> SIGMOID

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true 'label' vector (containing 0 if not-cat; 1 if cat) of shape (1, number of examples)
    layer_dims -- dimensions of the layers (n_x, n_h, n_y)
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
    parameters -- a dictionary containing W1, W2, b1 and b2
    '''
    np.random.seed(1)
    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layer_dims

    # Initialize the parameters
    parameters = initialize_parameters(n_x, n_h, n_y)

    # Get W1, b1, W2, b2 from the dictionary parameters
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    for i in range(0, num_iterations):
        # Forward propagation
        A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, 'sigmoid')
        
        #print('---2 Layer Model---A2[0]:', A2[0][:5])
        # Compute cost
        cost = compute_cost(A2, Y)

        dA2 = -(np.divide(Y, A2) - np.divide(1-Y, 1-A2))

        # Backward propagation
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        if i % 1 == 0:
            costs.append(np.squeeze(cost))
            if print_cost:
                print('---Two Layer Model---Cost after iteration {0}: {1}'.format(i, np.squeeze(cost)))

    return parameters



# L-layer Neural Network
def L_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    '''
    Implement a L-layer neural network: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true 'label' vector (containing 0 if not-cat; 1 if cat) of shape (1, number of examples)
    layer_dims -- list containing the input size and each layer size, of length (number of layers + 1)
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
    parameters -- parameters learnt by model. They can be used to predict.
    '''
    np.random.seed(1)
    costs = []

    parameters = initialize_parameters_deep(layer_dims)

    for i in range(0, num_iterations):
        # Forward Propagation
        AL, caches = L_model_forward(X, parameters)
        #print('---L Layer Model---AL[0]:', AL[0][:5])
        # Compute cost
        cost = compute_cost(AL, Y)
        #print('---L Layer Model---Cost after iteration {0}: {1}'.format(i, cost))
        # Backward Propagation
        grads = L_model_backward(AL, Y, caches)
        #print('---L Layer Model---dW2', grads['dW2'][0])
        #print('---L Layer Model---db2', grads['db2'][0])
        #print('---L Layer Model---dW1', grads['dW1'][0][:5])
        #print('---L Layer Model---db1', grads['db1'][0])
        # Update Parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        #print('---L Layer Model---After Update W2', parameters['W2'][0])
        #print('---L Layer Model---After Update b2', parameters['b2'][0])
        #print('---L Layer Model---After Update W1', parameters['W1'][0][:5])
        #print('---L Layer Model---After Update b1', parameters['b1'][0])

        if i % 100 == 0:
            costs.append(np.squeeze(cost))
            if print_cost:
                print('---L Layer Model---Cost after iteration {0}: {1}'.format(i, cost))
    return parameters



#layer_dims = (12288, 5, 1)
#parameters = two_layer_model(train_x, train_y, layer_dims, num_iterations=5, print_cost=True)


#layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
layers_dims = (12288, 5, 1)
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)


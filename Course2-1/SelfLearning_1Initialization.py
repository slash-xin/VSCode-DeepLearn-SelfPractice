# -*- encoding: utf-8 -*-
# Date:6 Mar 2018
# Author: Slash.Xin
# Descprition: Initialization - Improving Deep Neural Networks

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = load_dataset()
plt.show() # to show the pic from load_dataaset()

#-------------------------------------
# 1. Neural Network Model (3 layer)
#-------------------------------------
def model(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization='he'):
    '''
    Implement a 3-layer neural network: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.

    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true 'label' vector (0 for red dots; 1 for blue dots), of shape (1, number of examples)
    learning_rate -- learning rate for gradient descent
    num_iterations -- number of iterations to run gradient descent
    print_cost -- if True, print the cost every 1000 iterations
    initialization -- flag to choose which initialization to use ('zeros', 'random' or 'he')

    Returns:
    parameters -- parameters learnt by the model
    '''

    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 10, 5, 1]

    # Initialize parameters dictionary
    if initialization == 'zeros':
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == 'random':
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == 'he':
        parameters = initialize_parameters_he(layers_dims)
    
    for i in range(num_iterations):
        # Forward propagation
        A3, cache = forward_propagation(X, parameters)
        # Compute cost
        cost = compute_loss(A3, Y)
        # Backward propagation
        grads = backward_propagation(X, Y, cache)
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 1000 == 0:
            costs.append(cost)
            if print_cost:
                print('Cost after iteration {0}:{1}'.format(i, cost))
        
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('iterations (per thousands)')
    plt.title('Learning rate={}'.format(learning_rate))
    plt.show()

    return parameters


#-------------------------------------
# 2. Zero Initialization
#-------------------------------------
def initialize_parameters_zeros(layers_dims):
    '''
    Arguments:
    layers_dims -- python array (list) containing the size of each layer

    Returns:
    parameters -- python dictionary containing parameters 'W1', 'b1', ..., 'WL', 'bL'
    '''
    parameters = {}
    L = len(layers_dims)
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    
    return parameters

# Test the functions
parameters = initialize_parameters_zeros([3, 2, 1])
print("W1 = ", parameters['W1'])
print("b1 = ", parameters["b1"])
print("W2 = ", parameters["W2"])
print("b2 = ", parameters["b2"])

# Use parameters to train the model
parameters = model(train_X, train_Y, initialization='zeros')
print('On the train set:')
predictions_train = predict(train_X, train_Y, parameters)
print('On the test set:')
predictions_test = predict(test_X, test_Y, parameters)

# The performance is really bad, and the cost does not really decrease
# Look at the details of the predictions and the decision boundary.
print('predictions_train=', predictions_train)
print('predictions_test=', predictions_test)

plt.title("Model with Zeros initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y.ravel())

'''
The model is predicting 0 for every example. In general, initializing all the weights to zero results in the network failing to break symmetry.
This means that every neuron in each layer will learn the same thing, and you might as well be training a neural network with n = 1 for every layer,
and the network is no more powerful than a linear classifier such as logistic regression.
'''




#-------------------------------------
# 3. Random Initialization
#-------------------------------------
def initialize_parameters_random(layers_dims):
    '''
    Arguments:
    layers_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL".
    '''
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    
    return parameters

# Test the function
parameters = initialize_parameters_random([3, 2, 1])
print("W1 = ", parameters['W1'])
print("b1 = ", parameters["b1"])
print("W2 = ", parameters["W2"])
print("b2 = ", parameters["b2"])

# Use parameters to train the model
parameters = model(train_X, train_Y, initialization='random')
print('On the train set:')
predictions_train = predict(train_X, train_Y, parameters)
print('On the test set:')
predictions_test = predict(test_X, test_Y, parameters)

# The performance is really bad, and the cost does not really decrease
# Look at the details of the predictions and the decision boundary.
print('predictions_train=', predictions_train)
print('predictions_test=', predictions_test)

plt.title("Model with Zeros initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y.ravel())

'''
The cost starts very high. This is because with large random-valued weights, the last activation (sigmoid) 
outputs results that are very close to 0 or 1 for some examples, and when it gets that example wrong it 
incurs a very high loss for that example. 
Poor initialization can lead to vanishing/exploding gradients, which also slows down the optimization algorithm.
'''


#-------------------------------------
# 3. He Initialization
#-------------------------------------
'''
# Finally, try "He Initialization"; this is named for the first author of He et al., 2015. (If you have heard of "Xavier initialization", 
# this is similar except Xavier initialization uses a scaling factor for the weights W[l] of sqrt(1./layers_dims[l-1]) 
# where He initialization would use sqrt(2./layers_dims[l-1]).)
'''
def initialize_parameters_he(layers_dims):
    '''
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL".
    '''
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2./layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters

# Test the function
parameters = initialize_parameters_he([3, 2, 1])
print("W1 = ", parameters['W1'])
print("b1 = ", parameters["b1"])
print("W2 = ", parameters["W2"])
print("b2 = ", parameters["b2"])

# Use parameters to train the model
parameters = model(train_X, train_Y, initialization='he')
print('On the train set:')
predictions_train = predict(train_X, train_Y, parameters)
print('On the test set:')
predictions_test = predict(test_X, test_Y, parameters)

# The performance is really bad, and the cost does not really decrease
# Look at the details of the predictions and the decision boundary.
print('predictions_train=', predictions_train)
print('predictions_test=', predictions_test)

plt.title("Model with Zeros initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y.ravel())

'''
The model with He initialization separates the blue and the red dots very well in a small number of iterations.

Four Points:
Different initializations lead to different results.
Random initialization is used to break symmetry and make sure different hidden units can learn different things.
Don't intialize to values that are too large.
He initialization works well for networks with ReLU activations. 
'''
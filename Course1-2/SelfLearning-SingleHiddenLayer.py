# -*- encoding: utf-8 -*-
# Date: 2018.2.25 23:07:23
# Author: Slash.Xin
# Descprition: Single Hidden Layer Neural Network

# ----------------------
# 1. Package Import
# ----------------------
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from testCases import *
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

# ------------------------------------------------------------------------
# 2. Load DataSet: load a "flower" 2-class dataset into variables X and Y
# ------------------------------------------------------------------------
X, Y = load_planar_dataset()

# Visualize the data
plt.scatter(X[0, :], X[1, :], c=Y.squeeze(), s=40, cmap=plt.cm.Spectral)
plt.show()

# Get more details of data
print('shape of X:', X.shape)
print('shape of Y:', Y.shape)
print('number of traing examples:', X.shape[1])
m = X.shape[1] #training set size

# ------------------------------
# 3. Simple Logistic Regression
# ------------------------------
# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.ravel())

# Plot the dicision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y.ravel())
plt.title('Logistic Regression')
plt.show()
#Print the accuracy
LR_predictions = clf.predict(X.T)
print('Accuracy of logistic regression: {0} %'.format(float((np.dot(Y, LR_predictions) + np.dot(1-Y, 1-LR_predictions))/(Y.size)*100)))
'''
It turns out the dataset is not linearly separable, so logistic doesn't preform weell.
Hopefully a neural network will do better.
'''

# ------------------------
# 4. Neural Network Model
# ------------------------
# 4.1. Defining the neural network structor

def layer_sizes(X, Y):
    # size of input layer
    n_x = X.shape[0]
    # size of hidden layer
    n_h = 4
    # size of output layer
    n_y = Y.shape[0]

    return (n_x, n_h, n_y)

# Generate Test data, then test the function
X_assess, Y_assess = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
print('The size of the input layer is: {0}'.format(n_x))
print('The size of the hidden layer is: {0}'.format(n_h))
print('The size of the output layer is: {0}'.format(n_y))

# ----------------------------------------
# 4.2. Initialize the model's parameters
# ----------------------------------------
def initialize_parameters(n_x, n_h, n_y):

    # Set the random seed
    np.random.seed(2)

    # the first hidden layer
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    # the output layer
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    # Validate the dimension
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    parameters = {"W1":W1, "b1":b1, "W2":W2, "b2":b2}
    return parameters

# Generate the test data, then test the function
n_x, n_h, n_y = initialize_parameters_test_case()
parameters = initialize_parameters(n_x, n_h, n_y)
print('W1 =', parameters['W1'])
print('\nb1 =', parameters['b1'])
print('\nW2 =', parameters['W2'])
print('\nb2 =', parameters['b2'])


# ---------------
# 4.3. The Loop 
# ---------------

# Implement forward_propagation()
def forward_propagation(X, parameters):
    '''
    Arguments:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Results:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    '''

    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Implement forward propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    # Validate the dimension
    assert(A2.shape == (1, X.shape[1]))

    # Cache the values
    cache = {"Z1":Z1, "A1":A1, "Z2":Z2, "A2":A2}

    return A2, cache

# Generate the test data, then test the function
X_assess, parameters = forward_propagation_test_case()
A2, cache = forward_propagation(X_assess, parameters)
print('the mean value of Z1:', np.mean(cache['Z1']))
print('the mean value of A1:', np.mean(cache['A1']))
print('the mean value of Z2:', np.mean(cache['Z2']))
print('the mean value of A2:', np.mean(cache['A2']))

# Implement compute_cost() function to compute the value of the cost J
def compute_cost(A2, Y, parameters):
    '''
    Compute the cross-entropy cost given in equation

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameter W1, b1, W2, and b2

    Returns:
    cost -- cross-entropy cost given equation
    '''

    # Save the number of examples.
    m = Y.shape[1]

    # compute the cross-entropy cost
    logprobs = np.multiply(Y, np.log(A2)) + np.multiply(1-Y, np.log(1-A2))
    cost = -1.0 / m * np.sum(logprobs)
    cost = np.squeeze(cost)

    assert(isinstance(cost, float))

    return cost

# Generate the test data, then test the function
A2, Y_assess, parameters = compute_cost_test_case()
print('cost =', compute_cost(A2, Y_assess, parameters))


# Implement the function backward_propagation().
def backward_propagation(parameters, cache, X, Y):
    '''
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    '''

    # Save the number of examples.
    m = X.shape[1]

    # Retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters['W1']
    W2 = parameters['W2']
    # Retrieve A1 and A2 from dictionary "cache"
    A1 = cache["A1"]
    A2 = cache["A2"]

    #Backward propagation: calculate dw1, db1, dw2, db2
    dZ2 = A2 - Y
    dW2 = 1.0 / m * np.dot(dZ2, A1.T)
    db2 = 1.0 / m * np.sum(dZ2, axis=1, keepdims=True)
    # the derivative of tanh() is: 1 - a^2
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1.0 / m * np.dot(dZ1, X.T)
    db1 = 1.0 / m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1":db1, "dW2":dW2, "db2":db2}

    return grads

# Generate the test data, then test the function
parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print('dW1 =', grads['dW1'])
print('db1 =', grads['db1'])
print('dW2 =', grads['dW2'])
print('db2 =', grads['db2'])


# Gradient Function: update_parameters.
def update_parameters(parameters, grads, learning_rate=1.2):
    '''
    Updates parameters using the gradient descent update rule given aboce.
    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients
    Returns:
    parameters -- python dictionary containing your updated parameters
    '''

    # Retrieve each parameter from the dictionary 'parameters'
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Retrieve each gradient from the dictionary 'grads'
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    # Update rule for each parameter
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    parameters = {"W1":W1, "b1":b1, "W2":W2, "b2":b2}

    return parameters

# Generate test data, then test function
parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads)
print('W1 =', parameters['W1'])
print('b1 =', parameters['b1'])
print('W2 =', parameters['W2'])
print('b2 =', parameters['b2'])


# ----------------------------------------------------
# 4.4. Integrate parts 4.1, 4.2 and 4.3 in nn_model()
# ----------------------------------------------------
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    '''
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    '''

    np.random.seed(3)

    # Save the dimension of input data feature : input layer
    n_x = layer_sizes(X, Y)[0]
    # Save the dimension of output layer
    n_y = layer_sizes(X, Y)[2]

    # initialize the parameters
    parameters = initialize_parameters(n_x, n_h, n_y)
    #W1 = parameters['W1']
    #b1 = parameters['b1']
    #W2 = parameters['W2']
    #b2 = parameters['b2']

    # Loop: Gradient Descent
    for i in range(0, num_iterations):

        # Forward propagation. Inputs: "X, parameters"; Outputs: "A2, cache"
        A2, cache = forward_propagation(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters"; Outputs: "cost"
        cost = compute_cost(A2, Y, parameters)

        # Backward propagation. Inputs: "parameters, cache, X, Y"; Output: "grads"
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads"; Output: "parameters"
        parameters = update_parameters(parameters, grads)

        if print_cost and i % 1000 == 0:
            print('\nCost after iteration {0}: {1}'.format(i, cost))
        
    return parameters

# Generate test data, test the function
X_assess, Y_assess = nn_model_test_case()
parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=True)
print('W1 =', parameters['W1'])
print('b1 =', parameters['b1'])
print('W2 =', parameters['W2'])
print('b2 =', parameters['b2'])
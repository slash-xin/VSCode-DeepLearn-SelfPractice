# -*- encoding: utf-8 -*-
# Date: 2 Mar 2018
# Author: Slash.Xin
# Descprition: Deep Hidden Layer Neural Network Model Class. Use this class to construct a deep 
#              hidden layer neural network. It can only to be used binary classification.

import numpy as np

class DnnBinaryClassifierClass:
    '''
    Used to construct a deep hidden layer neural network. It can only to be used binary 
    classification. The size of hidden layer, the number of untis with each hidden layer,
    the activation function of each hidden layer can be set differently. And the activation 
    function of outpu layer is sigmoid which is can not changed.
    '''
    def __init__(self):
        '''
        Create the internal variables which are stored the information about the model.
        '''
        self.costs = []        # the cost of each 100 iterations.
        self.parameters = {}   # the matrix W and vector b of each layer. 
        self.grads = {}        # the gradient of W and b for each layer.
        self.caches = []       # keep the maily value to compute the gradient.

    def InitializeTrainData(self, train_x, train_y):
        '''
        Initialize the Training data.
        
        Argument:
        train_x -- the matrix of training features, shape: (number of features, number of samples).
        train_y -- the label of training data, shape (1, number of samples).
        '''

        if train_x.shape[1] != train_y.shape[1] or train_y.shape[0] != 1:
            infor = 'The shape of train_x and train_y are not matching. the shape of train_x must be (number of features, number of sample); the shape of train_y must be (1, number of samples).'
            raise RuntimeError('ShapeNotMatching', infor)
        else:
            self.train_x = train_x
            self.train_y = train_y
            self.train_data_size = train_x.shape[1]

    def InitializeModel(self, layer_dims, layer_activations):
        '''
        Initialize the model, include the layer size, the number of units of each layer, the activation function of each layer.

        Arguments:
        layer_dims -- list, the size of each hidden layer (exclude the input layer and output laeyer). The first element must be the number of units of first hidden layer, and so on.
                      For example, (8, 6, 5) means the model has 3 hidden layer; 1th hidden layer has 8 units, 2nd hidden layer has 6 units, 3rd hidden has 5 units.
        layer_activations -- list, the activation functions of each hidden layer (exclude the output layer cause it's alaways sigmoid). The first element must be the activation function
                             name of first hidden layer, and so on. For example, ('relu', 'tanh', 'relu') means 1th and 3rd hidden layer has RELU activation function, 2nd has TANH activation function.
        '''
        if len(layer_dims) != len(layer_activations):
            infor = 'The length of layer_dims must equal to layer_activations. layer_dims kepps the size of each hidden layer; layer_activations keeps the activation function of each hidden layer.'
            raise RuntimeError('DimensionNotEqual', infor)
        else:
            self.layer_dims = layer_dims.copy()
            self.layer_dims.insert(0, self.train_x.shape[0])
            self.layer_dims.append(1)
            self.layer_activations = layer_activations.copy()
            # number of layers (exclude the input layer, include the output layer)
            self.L = len(self.layer_dims) - 1
    
    def InitializeParameters(self):
        '''
        Initialize the matrix W and vector b of each layer (include hidden layers and output layer), for internal use.
        '''
        self.costs = []
        self.grads = {}
        self.caches = []

        np.random.seed(1)
        # Initialize the matrix W and b for hidden layers and output layer
        for l in range(1, self.L + 1):
            self.parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
            self.parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))
       
    def __sigmoid(self, Z):
        '''
        The Sigmoid activation function, for internal use.
        '''
        A = 1.0 / (1 + np.exp(-Z))
        return A
    def __relu(self, Z):
        '''
        The Relu activations function, for internal use.
        '''
        A = np.maximum(0, Z)
        return A
    def __tanh(self, Z):
        '''
        The Tanh activation function, for internal use.
        '''
        A = np.tanh(Z)
        return A
    
    def __activation_forward(self, A, W, b, activation):
        '''
        single step of forward propagation, for internal use.
        '''
        Z = np.dot(W, A) + b
        cache = ((A, W, b), Z)

        if activation == 'sigmoid':            
            A = self.__sigmoid(Z)            
        elif activation == 'relu':
            A = self.__relu(Z)
        elif activation == 'tanh':
            A = self.__tanh(Z)
        
        return A, cache

    
    def ForwardPropagation(self):
        '''
        The forward propagation.
        '''
        A = self.train_x
        # For every iteration of forward propagation, clear the caches.
        self.caches = []
        # Implement forward propagation for hidden layers
        for l in range(1, self.L):
            A_prev = A
            A, cache = self.__activation_forward(A_prev, self.parameters['W'+str(l)], self.parameters['b'+str(l)], self.layer_activations[l-1])
            self.caches.append(cache)

        # Implement forward propagation for output layer
        self.AL, cache = self.__activation_forward(A, self.parameters['W'+str(self.L)], self.parameters['b'+str(self.L)], 'sigmoid')
        self.caches.append(cache)
 
    def ComputeCost(self):
        '''
        Compute the cost of each iteration.
        '''
        cost = -1.0 / self.train_data_size * np.sum(np.multiply(self.train_y, np.log(self.AL)) + np.multiply((1-self.train_y), np.log(1-self.AL)), axis=1, keepdims=True)
        cost = np.squeeze(cost)
        return cost

    def __sigmoid_backward(self, dA, cache):
        '''
        the gradient of Sigmoid activation function, for internal use.
        '''
        Z = cache
        s = 1.0 / (1 + np.exp(-Z))
        dZ = dA * s * (1-s)
        return dZ
    def __relu_backward(self, dA, cache):
        '''
        the gradient of Relu activation function, for internal use.
        '''
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ
    def __tanh_backward(self, dA, cache):
        '''
        the gradient of Tanh activation function, for internal use.
        '''
        Z = cache
        t = np.tanh(Z)
        dZ = dA * (1.0 - np.power(t, 2))
        return dZ
    
    def __activation_backward(self, dA, cache, activation):
        '''
        single step of backward propagation, for internal use.
        '''
        linear_cache, activation_cache = cache

        if activation == 'relu':
            dZ = self.__relu_backward(dA, activation_cache)
        elif activation == 'sigmoid':
            dZ = self.__sigmoid_backward(dA, activation_cache)
        elif activation == 'tanh':
            dZ = self.__tanh_backward(dA, activation_cache)
        
        A_prev, W, b = linear_cache
        dW = 1.0 / self.train_data_size * np.dot(dZ, A_prev.T)
        db = 1.0 / self.train_data_size * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db
    
    def BackwardPropagation(self):
        '''
        The backward propagation.
        '''
        # derivative of cost function with AL
        dAL = -(np.divide(self.train_y, self.AL) - np.divide(1 - self.train_y, 1 - self.AL))

        current_cache = self.caches[self.L - 1]
        self.grads['dA' + str(self.L)], self.grads['dW'+str(self.L)], self.grads['db'+str(self.L)] = self.__activation_backward(dAL, current_cache, 'sigmoid')
        
        for l in reversed(range(self.L-1)):
            current_cache = self.caches[l]
            self.grads['dA'+str(l+1)], self.grads['dW'+str(l+1)], self.grads['db'+str(l+1)] = self.__activation_backward(self.grads['dA'+str(l+2)], current_cache, self.layer_activations[l])

    def UpdateParameters(self, learning_rate):
        '''
        Use gradicent to update the parameters W and b.
        '''
        # number of layers (exclude input layer)
        for l in range(self.L):
            self.parameters['W' + str(l+1)] -= learning_rate * self.grads['dW' + str(l+1)]
            self.parameters['b' + str(l+1)] -= learning_rate * self.grads['db' + str(l+1)]
    
    def TrainModel(self, learning_rate=0.0075, num_iterations=2500, print_cost=False):
        '''
        Train the model.
        Arguments:
        learning_rate -- the learning rate.
        num_iterations -- the total iterations.
        print_cost -- If set to True, this will print the cost every 100 iterations
        '''
        # Initialize the parameters
        self.InitializeParameters()

        for i in range(num_iterations):
            self.ForwardPropagation()
            cost = self.ComputeCost()
            self.BackwardPropagation()
            self.UpdateParameters(learning_rate)

            if i % 100 == 0:
                self.costs.append(np.squeeze(cost))
                if print_cost:
                    print('---{0} Layer Model---Cost after iteration {0}: {1}'.format(self.L, i, cost))

    def Predict(self, X):
        '''
        Predict the results of specified X.
        '''
        m = X.shape[1]
        p = np.zeros((1, m))

        A = X
        for l in range(1, self.L):
            A_prev = A
            A, cache = self.__activation_forward(A_prev, self.parameters['W'+str(l)], self.parameters['b'+str(l)], self.layer_activations[l-1])

        # Implement forward propagation for output layer
        probas, cache = self.__activation_forward(A, self.parameters['W'+str(self.L)], self.parameters['b'+str(self.L)], 'sigmoid')

        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0
        return p
    
    def GetCosts(self):
        '''
        Retrieve the cost of each 100 iteration.
        Returns:
        costs -- list, include the cost of every 100 iteration.
        '''
        return self.costs
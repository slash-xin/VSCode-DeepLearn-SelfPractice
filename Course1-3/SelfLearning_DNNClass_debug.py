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
        self.costs = []
        self.parameters = {}
        self.grads = {}
        self.caches = []

    def InitializeTrainData(self, train_x, train_y):
        '''
        '''
        self.train_x = train_x
        self.train_y = train_y
        self.train_data_size = train_x.shape[1]

    def InitializeModel(self, layer_dims, layer_activations):
        '''
        '''
        self.layer_dims = layer_dims
        self.layer_activations = layer_activations
        # number of layers (exclude the input layer)
        self.L = len(layer_dims) - 1
    
    def InitializeParameters(self):
        '''
        '''
        np.random.seed(1)
        # Initialize the matrix W and b for hidden layers and output layer
        for l in range(1, self.L + 1):
            self.parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
            self.parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))
       
    def __sigmoid(self, Z):
        A = 1.0 / (1 + np.exp(-Z))
        return A
    def __relu(self, Z):
        A = np.maximum(0, Z)
        return A
    def __tanh(self, Z):
        A = np.tanh(Z)
        return A
    
    def __activation_forward(self, A, W, b, activation):
        '''
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
        A = self.train_x
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
        '''
        cost = -1.0 / self.train_data_size * np.sum(np.multiply(self.train_y, np.log(self.AL)) + np.multiply((1-self.train_y), np.log(1-self.AL)), axis=1, keepdims=True)
        cost = np.squeeze(cost)
        return cost

    def __sigmoid_backward(self, dA, cache):
        Z = cache
        s = 1.0 / (1 + np.exp(-Z))
        dZ = dA * s * (1-s)
        return dZ
    def __relu_backward(self, dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ
    def __tanh_backward(self, dA, cache):
        Z = cache
        t = np.tanh(Z)
        dZ = dA * (1.0 - np.power(t, 2))
        return dZ
    
    def __activation_backward(self, dA, cache, activation):
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

        #print('===================== shape of A_prev:', A_prev.shape)
        #print('===================== shape of dZ:', dZ.shape)        
        #print('===================== shape of dW:', dW.shape)
        #print('===================== shape of db:', db.shape)
        #print('===================== shape of dA_prev:', dA_prev.shape)

        return dA_prev, dW, db
    
    def BackwardPropagation(self):
        # derivative of cost function with AL
        dAL = -(np.divide(self.train_y, self.AL) - np.divide(1 - self.train_y, 1 - self.AL))

        #print('=====================BackwardPropagation: AL:', self.AL[0][:5])
        #print('=====================BackwardPropagation: Y:', self.train_y[0][:5])
        #print('=====================BackwardPropagation: dAL:', dAL[0][:5])

        current_cache = self.caches[self.L - 1]
        #print('==================Current Cache============', current_cache[0][0][:5])
        self.grads['dA' + str(self.L)], self.grads['dW'+str(self.L)], self.grads['db'+str(self.L)] = self.__activation_backward(dAL, current_cache, 'sigmoid')

        #print('=====================BackwardPropagation: dA{0}, shape:{1}'.format(self.L, self.grads['dA'+str(self.L)].shape))
        #print('=====================BackwardPropagation: dW{0}, shape:{1}'.format(self.L, self.grads['dW'+str(self.L)].shape))
        #print('=====================BackwardPropagation: db{0}, shape:{1}'.format(self.L, self.grads['db'+str(self.L)].shape))
        
        for l in reversed(range(self.L-1)):
            current_cache = self.caches[l]
            self.grads['dA'+str(l+1)], self.grads['dW'+str(l+1)], self.grads['db'+str(l+1)] = self.__activation_backward(self.grads['dA'+str(l+2)], current_cache, self.layer_activations[l])

            #print('=====================BackwardPropagation: dA{0}, shape:{1}'.format(l+1, self.grads['dA'+str(l+1)].shape))
            #print('=====================BackwardPropagation: dW{0}, shape:{1}'.format(l+1, self.grads['dW'+str(l+1)].shape))
            #print('=====================BackwardPropagation: db{0}, shape:{1}'.format(l+1, self.grads['db'+str(l+1)].shape))

    def UpdateParameters(self, learning_rate):
        # number of layers (exclude input layer)
        for l in range(self.L):
            #print('---------------shape of W{0}: {1}'.format(l+1, self.parameters['W'+str(l+1)].shape))
            #print('---------------shape if dW{0}: {1}'.format(l+1, self.grads['dW'+str(l+1)].shape))
            self.parameters['W' + str(l+1)] -= learning_rate * self.grads['dW' + str(l+1)]
            self.parameters['b' + str(l+1)] -= learning_rate * self.grads['db' + str(l+1)]
    
    def TrainModel(self, learning_rate=0.0075, num_iterations=2500, print_cost=False):
        # Initialize the parameters
        self.InitializeParameters()

        for i in range(num_iterations):
            self.ForwardPropagation()
            #print('---L Layer Model---AL[0]:', self.AL[0][:5])
            cost = self.ComputeCost()
            #print('---L Layer Model---Cost after iteration {0}: {1}'.format(i, cost))
            self.BackwardPropagation()
            #print('---L Layer Model---dW2', self.grads['dW2'][0])
            #print('---L Layer Model---db2', self.grads['db2'][0])
            #print('---L Layer Model---dW1', self.grads['dW1'][0][:5])
            #print('---L Layer Model---db1', self.grads['db1'][0])
            self.UpdateParameters(learning_rate)
            #print('---L Layer Model---After Update W2', self.parameters['W2'][0])
            #print('---L Layer Model---After Update b2', self.parameters['b2'][0])
            #print('---L Layer Model---After Update W1', self.parameters['W1'][0][:5])
            #print('---L Layer Model---After Update b1', self.parameters['b1'][0])

            if i % 100 == 0:
                self.costs.append(np.squeeze(cost))
                if print_cost:
                    print('---L Layer Model---Cost after iteration {0}: {1}'.format(i, cost))
            
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
        self.train_data_size = train_x[1]

    def InitializeModel(self, layer_dims, layer_activations):
        '''
        '''
        self.layer_dims = layer_dims
        self.layer_activations = layer_activations
    
    def InitializeParameters(self):
        '''
        '''
        L = len(self.layer_dims)
        for l in range(1, L):
            self.parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
            self.parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))
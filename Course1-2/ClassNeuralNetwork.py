# -*- encoding: utf-8 -*-
'''
Description: used to train a single hidden layer neural network for binary classification.
Date: 27 Feb 2018
Author: Slash.Xin
'''

import numpy as np

class SingleHiddenLayer:
    def __init__(self, n_h, learning_rate=1.2):
        self.n_h = n_h
        self.learning_rate = learning_rate
    
    def initializeData(self, X, Y):
        self.X = X
        self.Y = Y
    
    def initialize_parameters(self):
        np.random.seed(2)

        n_x = self.X.shape[0]
        n_y = self.Y.shape[0]

        W1 = np.random.randn(self.n_h, n_x) * 0.01
        b1 = np.zeros(shape=(self.n_h, 1))
        W2 = np.random.randn(n_y, self.n_h) * 0.01
        b2 = np.zeros(shape=(n_y, 1))

        self.parameters = {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2}
    
    def sigmoid(self, x):
        s = 1.0 / (1 + np.exp(-x))
        return s
    
    def forward_propagation(self):
        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']

        Z1 = np.dot(W1, self.X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(Z2)

        self.cache = {'Z1':Z1, 'A1':A1, 'Z2':Z2, 'A2':A2}
    
    def compute_cost(self):
        m = self.Y.shape[1]

        logprobs = np.multiply(self.Y, np.log(self.cache['A2'])) + np.multiply(1-self.Y, np.log(1-self.cache['A2']))
        cost = -1.0 / m * np.sum(logprobs)

        self.cost = np.squeeze(cost)
    
    def backward_propagation(self):
        m = self.Y.shape[1]

        W2 = self.parameters['W2']
        A1 = self.cache['A1']
        A2 = self.cache['A2']

        dZ2 = A2 - self.Y
        dW2 = 1.0 / m * np.dot(dZ2, A1.T)
        db2 = 1.0 / m * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = 1.0 / m * np.dot(dZ1, self.X.T)
        db1 = 1.0 / m * np.sum(dZ1, axis=1, keepdims=True)

        self.grads = {'dW1': dW1, 'db1':db1, 'dW2':dW2, 'db2':db2}
    
    def update_parameters(self):
        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']

        W1 -= self.learning_rate * self.grads['dW1']
        b1 -= self.learning_rate * self.grads['db1']
        W2 -= self.learning_rate * self.grads['dW2']
        b2 -= self.learning_rate * self.grads['db2']

        self.parameters = {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2}
    
    def train_model(self, num_iterations=10000):

        np.random.seed(3)

        self.initialize_parameters()

        for i in range(0, num_iterations):
            self.forward_propagation()
            self.compute_cost()
            self.backward_propagation()
            self.update_parameters()

            if i % 1000 == 0:
                print('Cost after iteration {0}: {1}'.format(i, self.cost))
    
    def predict(self, X):
        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']

        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(Z2)

        predictions = np.around(A2)

        return predictions
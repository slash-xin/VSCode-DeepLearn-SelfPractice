# -*- encoding: utf-8 -*-
'''
Description: Test My Model.
Date: 27 Feb 2018
Author: Slash.Xin
'''
import numpy as np
import matplotlib.pyplot as plt
from ClassNeuralNetwork import SingleHiddenLayer
from planar_utils import load_planar_dataset, load_extra_datasets, plot_decision_boundary

def Test1():
    X, Y = load_planar_dataset()

    model = SingleHiddenLayer(4, 1.2)
    model.initializeData(X, Y)
    model.train_model(num_iterations=10000)
    predictions = model.predict(X)
    print ('Accuracy: {0}%'.format(float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)))
    plot_decision_boundary(lambda x: model.predict(x.T), X, Y.ravel())
    plt.title('Decision boundary for hidden layer size: 5')
    plt.show()

Test1()


def Test2():
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
    datasets = {'noisy_circles':noisy_circles, 'noisy_moons':noisy_moons, 'blobs':blobs, 'gaussian_quantiles':gaussian_quantiles}
    # choose one dataset
    dataset = 'gaussian_quantiles'
    X, Y = datasets[dataset]
    X, Y = X.T, Y.reshape(1, Y.shape[0])

    print(X.shape, Y.shape)

    # make blobs binary
    if dataset == 'blobs':
        Y = Y % 2

    # Visualize the data
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.show()

    model = SingleHiddenLayer(5, 0.2)
    model.initializeData(X, Y)
    model.train_model(num_iterations=10000)
    predictions = model.predict(X)
    print ('Accuracy: {0}%'.format(float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)))
    plot_decision_boundary(lambda x: model.predict(x.T), X, Y.ravel())
    plt.title('Decision boundary for hidden layer size: 5')
    plt.show()

Test2()
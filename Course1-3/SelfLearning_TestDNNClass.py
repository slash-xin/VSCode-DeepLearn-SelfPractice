
import numpy as np
import h5py
from SelfLearning_DNNClass import DnnBinaryClassifierClass

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


layer_dims = (12288, 5, 1)
layer_activations = ['relu']

model = DnnBinaryClassifierClass()
model.InitializeTrainData(train_x, train_y)
model.InitializeModel(layer_dims, layer_activations)
model.TrainModel(num_iterations=2500, print_cost=True)
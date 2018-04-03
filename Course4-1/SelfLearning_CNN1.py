# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------
   File Name：    SelfLearning_CNN1
   Description :  Learning the CNN
   Author :       slash
   Date：         3/31/2018
--------------------------------------------------------------
   Change Activity:
      3/31/2018: First Version
--------------------------------------------------------------
"""
__author__ = 'slash'

# ---------------------------------
# 1. Import the package
# ---------------------------------

import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [5.0, 4.0]
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# ---------------------------------
# 2. Convolutional Neural Networks
# ---------------------------------
# 2.1. Zero-Padding
def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image.

    :param X: python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images.
    :param pad: integer, amount of padding around each image on vertical and horizontal dimensions.
    :return X_pad: padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """

    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
    return X_pad

# Test the function
x = np.random.randn(4, 3, 2, 2)
x_pad = zero_pad(x, 2)
print('x.shape =', x.shape)
print('x_pad.shape', x_pad.shape)
print('x[1, 1] =', x[1, 1])
print('x_pad[1, 1] =', x_pad[1, 1])

fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0, :, :, 0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0, :, :, 0])
plt.show()



# ---------------------------------
# 2.2. Single step of convolution
# ---------------------------------
def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prec) of  the output activation
    of the previous layer.

    :param a_slice_prev: slice of input data of shape (f, f, n_C_prev)
    :param W: Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    :param b: Bias parameters contained in a window - matrix of shape (1, 1, 1)
    :return Z: a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """
    s = a_slice_prev * W + b
    Z = np.sum(s, keepdims=True)
    return Z

# Test the function
np.random.seed(1)
a_slice_prev = np.random.randn(4, 4, 3)
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)

Z = conv_single_step(a_slice_prev, W, b)
print('Z =', Z)



# --------------------------------------------------
# 2.3. Convolutional Neural Networks - Forward pass
# --------------------------------------------------
def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function

    :param A_prev: output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    :param W: Weights, numpy array of shape (f, f, n_C_prev, n_C)
    :param b: Biases, numpy array of shape (1, 1, 1, n_C)
    :param hparameters: python dictionary containing "stride" and "pad"
    :return Z: conv output, numpy array of shape (m, n_H, n_W, n_C)
    :return cache: cache of values needed for the con_backward() function
    """

    # Retrieve dimensions from A_prev and W
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from hparameters
    stride = hparameters['stride']
    pad = hparameters['pad']

    # Compute the dimensions of the CONV output volume.
    n_H = int(np.floor((n_H_prev - f + 2 * pad) / stride) + 1)
    n_W = int(np.floor((n_W_prev - f + 2 * pad) / stride) + 1)

    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i, :, :, :]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # Find the corners of the current 'slice'
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f
                    # Use the corners to define 3D slice of a_prev_pad
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])

    assert(Z.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, W, b, hparameters)

    return Z, cache

# Test the function
np.random.seed(1)
A_prev = np.random.randn(10, 4, 4, 3)
W = np.random.randn(2, 2, 3, 8)
b = np.random.randn(1, 1, 1, 8)
hparameters = {"pad":2, "stride":1}

Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
print("Z's mean =", np.mean(Z))
print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])
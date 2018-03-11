#------------------------------------------------
# 2. Build first neural network with tensorflow
#------------------------------------------------

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict


# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Example of a picture
index = 0
plt.imshow(X_train_orig[index])
plt.show()
print('y = {}'.format(np.squeeze(Y_train_orig[:, index])))



# Flattern the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

# Normalize image vectors
X_train = X_train_flatten / 255.0
X_test = X_test_flatten / 255.0

# Convert training and test labels to one hot matrix
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

print('number of training example:', X_train.shape[1])
print('number of test examples:', X_test.shape[1])
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)


#-------------------------
# 2.1. Create placeholder
#-------------------------
def create_placeholder(n_x, n_y):
    '''
    Create the placeholder for the tensorflow session.

    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of calsses (from 0 to 5, so -> 6)

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype 'float'
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype 'float'

    Tips:
    - Use None because the number of examples during test/train is different.
    '''
    X = tf.placeholder(shape=[n_x, None], dtype=tf.float32)
    Y = tf.placeholder(shape=[n_y, None], dtype=tf.float32)

    return X, Y

# Test the function
X, Y = create_placeholder(12288, 6)
print('X =', X)
print('Y =', Y)


#-----------------------------------
# 2.2. Initializing the parameters
#-----------------------------------
def Initializing_parameters():
    '''
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    '''
    tf.set_random_seed(1)

    W1 = tf.get_variable('W1', [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable('b1', [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable('W2', [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable('b2', [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable('W3', [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable('b3', [6, 1], initializer=tf.zeros_initializer())

    parameters = {'W1':W1, 'b1':b1,
                  'W2':W2, 'b2':b2,
                  'W3':W3, 'b3':b3}

    return parameters

# Test the function
tf.reset_default_graph()
with tf.Session() as sess:
    parameters = Initializing_parameters()
    print('W1 =', parameters['W1'])
    print('b1 =', parameters['b1'])
    print('W2 =', parameters['W2'])
    print('b2 =', parameters['b2'])
    print('W3 =', parameters['W3'])
    print('b3 =', parameters['b3'])


#----------------------------------------
# 2.3. Forward propagation in tensorflow
#----------------------------------------
def forward_propagation(X, parameters):
    '''
    Implements the forward propagation for the model.

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    '''
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3

# Test the funcion
tf.reset_default_graph()
with tf.Session() as sess:
    X, Y = create_placeholder(12288, 6)
    parameters = Initializing_parameters()
    Z3 = forward_propagation(X, parameters)
    print('Z3 =', Z3)


#----------------------------------------
# 2.4. Compute cost
#----------------------------------------
def compute_cost(Z3, Y):
    '''
    Computes the cost

    Arguments:
    Z3 -- ouput of forward propagatoin (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- 'true' labels vector placeholder, same shape as Z3

    Returns:
    cost -- Tensor of the cost function
    '''
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost

# Test the fucntion
tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholder(12288, 6)
    parameters = Initializing_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    print('cost =', cost)


#------------------------------------------------
# 2.5. Backward propagation & parameters updates
#------------------------------------------------
#All the backpropagation and the parameters update is taken care of in 1 line of code. It is very easy to incorporate this line in the model.

#------------------------------------------------
# 2.6. Building the model
#------------------------------------------------
def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001, num_epoches=1500, minibatch_size=32, print_cost=True):
    '''
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    '''

    # to be able to rerun the model without overwriting tf variables
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    # Create placeholder
    X, Y = create_placeholder(n_x, n_y)

    # Initialize parameters
    parameters = Initializing_parameters()

    # Forward propagation
    Z3 = forward_propagation(X, parameters)

    # Cost function
    cost = compute_cost(Z3, Y)

    # Backward propagation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epoches):
            epoch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches
            
            if print_cost and epoch % 100 == 0:
                print('Cost after epoch {}: {}'.format(epoch, epoch_cost))
            if print_cost and epoch % 5 == 0:
                costs.append(epoch_cost)
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title('Learning rate={}'.format(learning_rate))
        plt.show()

        # save parameters in a variable
        parameters = sess.run(parameters)
        print('Parameters have been trained!')

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        print('Train Accuracy:', accuracy.eval({X:X_train, Y:Y_train}))
        print('Test Accuracy:', accuracy.eval({X:X_test, Y:Y_test}))

        return parameters

parameters = model(X_train, Y_train, X_test, Y_test)





import scipy
from PIL import Image
from scipy import ndimage

## START CODE HERE ## (PUT YOUR IMAGE NAME) 
my_image = "thumbs_up.jpg"
## END CODE HERE ##

# We preprocess your image to fit your algorithm.
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
my_image_prediction = predict(my_image, parameters)

plt.imshow(image)
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
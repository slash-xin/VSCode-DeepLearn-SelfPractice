# -*- encoding: utf-8 -*-
# Date: 9 Mar 2018
# Author: Slash.Xin
# Descprition: Tensoeflow Execrise


import numpy as np
import tensorflow as tf


#coeffocoents = np.array([[1.], [-10.], [25.]])
coeffocoents = np.array([[1.], [-20.], [100.]])

W = tf.Variable(0, dtype=tf.float32)
x = tf.placeholder(tf.float32, [3, 1])
#cost = tf.add(tf.add(W**2, tf.multiply(-10., W)), 25)
#cost = W**2 - 10*W + 25
cost = x[0][0]*W**2 + x[1][0]*W + x[2][0]
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print('Value of W:', session.run(W))


session.run(train, feed_dict={x:coeffocoents})
print('Value of W:', session.run(W))

for i in range(1000):
    session.run(train, feed_dict={x:coeffocoents})
print('Value of W:', session.run(W))


#----------------------------------------
# 1. Exploring the Tensorflow Library
#----------------------------------------
import numpy as np
import tensorflow as tf

np.random.seed(1)


# Define a loss to compute it.
y_hat = tf.constant(36, name='y_hat')
y = tf.constant(39, name='y')
loss = tf.Variable((y - y_hat) ** 2, name='loss')

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    print(session.run(loss))

# easy example
a = tf.constant(2)
b = tf.constant(10)
c =tf.multiply(a, b)
print(c)

sess = tf.Session()
print(sess.run(c))

# Placeholder
x = tf.placeholder(tf.int64, name='x')
print(sess.run(2 * x, feed_dict={x:3}))
sess.close()


#----------------------
# 1.1. Linear function
#----------------------
def linear_function():
    np.random.seed(1)

    X = tf.constant(np.random.randn(3, 1), name='X')
    W = tf.constant(np.random.randn(4, 3), name='W')
    b = tf.constant(np.random.randn(4, 1), name='b')
    Y = tf.add(tf.matmul(W, X), b)

    sess = tf.Session()
    result = sess.run(Y)

    sess.close()

    return result

print( "result = " + str(linear_function()))


#----------------------------
# 1.2. Computing the sigmoid
#----------------------------
def sigmoid(z):
    x = tf.placeholder(tf.float32, name='x')
    sigmoid = tf.sigmoid(x)

    with tf.Session() as sess:
        result = sess.run(sigmoid, feed_dict={x:z})
        
    return result

print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(12) = " + str(sigmoid(12)))


#----------------------------
# 1.3. Compute the Cost
#----------------------------
def cost(logits, labels):
    z = tf.placeholder(tf.float32, name='z')
    y = tf.placeholder(tf.float32, name='y')

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)

    sess = tf.Session()
    cost = sess.run(cost, feed_dict={z:logits, y:labels})
    sess.close()

    return cost

logits = sigmoid(np.array([0.2,0.4,0.7,0.9]))
cost = cost(logits, np.array([0,0,1,1]))
print ("cost = " + str(cost))


#-----------------------------
# 1.4. Using One-hot Encoding
#-----------------------------
def one_hot_matrix(labels, C):
    C = tf.constant(C, name='C')
    one_hot_matrix = tf.one_hot(labels, C, axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()

    return one_hot

labels = np.array([1,2,3,0,2,1])
one_hot = one_hot_matrix(labels, C = 4)
print ("one_hot = " + str(one_hot))

#-------------------------------------
# 1.5. Initialize with zeros and ones
#-------------------------------------
def ones(shape):
    ones = tf.ones(shape=shape, name='ones')
    sess = tf.Session()
    ones = sess.run(ones)
    sess.close()

    return ones
print ("ones = " + str(ones([3])))
print ("ones = " + str(ones([3, 2])))




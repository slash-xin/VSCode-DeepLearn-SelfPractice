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
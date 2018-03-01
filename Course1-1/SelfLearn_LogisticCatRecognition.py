# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:20:01 2018

@author: Slash
"""

import numpy as np                #使用Python进行科学计算的基础包
import matplotlib.pyplot as plt   #Python中著名的绘图库
import h5py                       #Python提供读取HDF5二进制数据格式文件的接口，本次的训练及测试图片集是以HDF5储存的
import os
import scipy                      #基于NumPy来做高等数学、信号处理、优化、统计和许多其它科学任务的拓展库
from PIL import Image             #(Python Image Library)为Python提供图像处理功能
from scipy import ndimage

#数据导入
def load_dataset(path):
    train_dataset = h5py.File(os.path.join(path, 'train_catvnoncat.h5'), 'r') #读取训练数据，共209张图片
    test_dataset = h5py.File(os.path.join(path, 'test_catvnoncat.h5'), 'r')   #读取测试数据，共50张图片
    
    train_set_x_orig = np.array(train_dataset['train_set_x'][:]) #原始训练集（209*64*64*3）
    train_set_y_orig = np.array(train_dataset['train_set_y'][:]) #原始训练集的标签集（y=0非猫，y=1猫）（209*1）
    
    test_set_x_orig = np.array(test_dataset['test_set_x'][:])
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])
    
    train_set_y_orig = train_set_y_orig.reshape(1, train_set_y_orig.shape[0]) #原始训练集标签的维度更改为（1*209）
    test_set_y_orig = test_set_y_orig.reshape(1, test_set_y_orig.shape[0])
    
    classes = np.array(test_dataset['list_classes'][:])
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

#加载数据
img_path = 'D:/JupyterNotebook/DeepLearning/deeplearning.ai-master/course1/assignment1-2/assignment2/datasets'
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset(img_path)


#显示图片
def image_show(index, dataset):
    if dataset == 'train':
        plt.imshow(train_set_x_orig[index])
        print('y = {0}, 它是一张{1}图片'.format(str(train_set_y[:, index]), classes[np.squeeze(train_set_y[:, index])].decode('utf-8')))
    elif dataset == 'test':
        plt.imshow(test_set_x_orig[index])
        print('y = {0}, 它是一张{1}图片'.format(str(test_set_y[:, index]), classes[np.squeeze(test_set_y[:, index])].decode('utf-8')))
#测试显示
image_show(14, 'train')
image_show(14, 'test')


#计算数据维度信息
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = test_set_x_orig.shape[1]

#将照片数据的维度展平
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T
print(train_set_x_flatten.shape)
print(test_set_x_flatten.shape)

#数据标准化
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.


#sigmoid函数
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s
#测试函数
print('sigmoid of 16:', sigmoid(16))


#初始化参数w, b
def initialize_with_zero(dim):
    w = np.zeros(shape=(dim, 1), dtype=float) #w为一个dim*1的矩阵
    b = 0.0
    return w, b
#测试函数
w, b = initialize_with_zero(5)
print(w, b)


#正向传播函数：计算Y_hat, 成本函数以及dw, db
def propagate(w, b, X, Y):
    #保存样本量
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1 * (np.sum(np.dot(Y, np.log(A).T) + np.dot((1-Y), np.log(1-A).T)))/m
    
    #X:(n,m); A:(1, m)
    dw = 1.0 * np.dot(X, (A-Y).T) / m
    db = 1.0 * np.sum(A-Y, axis=1, keepdims=True) / m
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    #cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {'dw':dw, 'db':db}
    
    return grads, cost
#测试函数
w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
grads, cost = propagate(w, b, X, Y)


#优化器：根据梯度值更新参数
def optimizer(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        
        dw = grads['dw']
        db = grads['db']
        
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print('Cost after iteration {0}: {1}'.format(i, cost))
        
    params = {'w':w, 'b':b}
    grads = {'dw':dw, 'db':db}
    
    return params, grads, costs
#测试函数
params, grads, costs = optimizer(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=True)
print('w=', str(params['w']))
print('b=', str(params['b']))
print('dw=', str(grads['dw']))
print('db=', str(grads['db']))


#定义预测函数
def predict(w, b, X):
    m = X.shape[1]
    
    Y_prediction = np.zeros((1,m), dtype=int)
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T, X) + b)
    print('shape of A:', A.shape)
    
    for i in range(A.shape[1]):
        if(A[0, i] > 0.5):
            Y_prediction[0, i] = 1
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction
#测试函数
print('Predictions = ', str(predict(w, b, X)))




#-----------------------------------------------
# Merge All Functions into a Model Function
#-----------------------------------------------
def CatRecognitionModel(X_train, Y_train, X_test, Y_test, num_iterations=200, learning_rate=0.5, print_cost=False):
    #Save the number of train samples.
    #m_train = X_train.shape[1]
    #Save the dimensions of feature of samples
    dim = X_train.shape[0]
    #Initialize the parameters: w, b
    w, b = initialize_with_zero(dim)

    #Gradient descent
    params, grads, costs = optimizer(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    #Retrieve the final parameters w and b
    w = params['w']
    b = params['b']

    #Predict the train/test samples
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    #Print train/test errors
    print('train accuracy: {0}'.format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print('test accuracy:{0}'.format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {'costs': costs,
    	'Y_prediction_test':Y_prediction_test,
    	'Y_prediction_train':Y_prediction_train,
    	'w':w,
    	'b':b,
    	'learning_rate':learning_rate,
    	'num_iterations':num_iterations}

    return d

#Train the Model
d = CatRecognitionModel(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)



# Check the Result
index =19
plt.imshow(test_set_x[:, index].reshape(num_px, num_px, 3))
print('y={0}, you predicted that it is a {1}.'.format(test_set_y[0, index], classes[d['Y_prediction_test'][0, index]].decode()))


#Plot cost with iteration
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title('Learning rate = {0}'.format(d['learning_rate']))
plt.show()

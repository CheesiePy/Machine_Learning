# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 19:49:20 2023

class
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def plot_reg_line(X, y, theta):
    """
    plot_reg_line plots the data points and regression line 
    for linear regrssion
    Input arguments: X - np array (m, n) - independent variable.
    y - np array (m,1) - target variable
    theta - parameters
    """
    if X.shape[1] == 2: 
        ind = 1
    else:
        ind = 0           
        
    x_min = X[:,ind].min()
    x_max = X[:,ind].max()
    ind_min = X[:,ind].argmin()
    ind_max = X[:,ind].argmax()
    y_min = y[ind_min]
    y_max = y[ind_max]
    Xlh = X[(ind_min, ind_max),:]
    yprd_lh = np.dot(Xlh, theta)
    plt.plot(X[:,ind], y, 'go', Xlh, yprd_lh, 'm-')
    plt.axis((x_min-5, x_max+5, min(y_min,y_max) - 5 , max(y_min,y_max)+5))
    plt.xlabel('x'), plt.ylabel('y'), 
    plt.title('Regression data')
    plt.grid()
    plt.show()
    
def computeCost(X, y, theta):
    m = y.size
    z = np.dot(X, theta) - y
    J = 1 / (2 * m) * np.dot(z.T, z)
    return J

def gd_ol(X, y, theta, alpha, num_iter):
    m = y.shape[0]
    J_iter = np.zeros((m * num_iter))
    k = 0
    for j in range(num_iter):
        randindex = np.random.permutation(m)
        for i in range(m):
            xi = X[randindex[i], :]
            xi = xi.reshape(1, xi.shape[0])
            yi = y[randindex[i]]
            delta = np.dot(xi, theta) - yi
            theta = theta - alpha * delta * xi.T
            J = computeCost(xi, yi, theta)
            J_iter[k] = J
            k = k + 1
        
            
    
    
    
    return theta, J_iter
    

data = np.load('Materials for ex. 1 - Linear Regression and Gradient Descent-20230322/Cricket.npz')
sorted(data)
yx = data['arr_0']
x = yx[:, 1]
y = yx[:, 0]
x = x.reshape(x.shape[0], 1)
y = y.reshape(y.shape[0], 1)
plt.plot(x, y, 'ro')
plt.grid(axis = 'both')
plt.show()
m = y.shape[0]
print('number of examples = ', m )
# ind_xmin = x.argmin()
# ind_xmax = x.argmax()
# ind_ymin = y.argmin()
# ind_ymax = y.argmax()
# xmin = x[ind_xmin]
# xmax = x[ind_xmax]
# ymin = y[ind_xmin]
# ymax = y[ind_xmax]
# xlh = np.array([xmin, xmax])
onesvec = np.ones((m, 1))
X = np.concatenate((onesvec, x), axis = 1)
n = X.shape[1]
#Xlh= X[(ind_xmin, ind_xmax), :]
alpha = 0.0001
theta = np.zeros((n, 1))
num_iter = 18
J = computeCost(X, y, theta)
print('J = ', J)
theta, J_iter = gd_ol(X, y, theta, alpha, num_iter)
plot_reg_line(X, y, theta)
plt.plot(J_iter)
plt.show()




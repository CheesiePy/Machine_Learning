# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 19:29:55 2023


"""
import numpy as np
import matplotlib.pyplot as plt
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
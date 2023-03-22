# Gradient Descent


# example 1 - simple gradient descent


import numpy as np
import matplotlib.pyplot as plt

# create data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10])

# plot data
plt.scatter(x, y)
plt.show()  

# calculate the mean of x and y
x_mean = np.mean(x)
y_mean = np.mean(y)

# total number of values
n = len(x)

# using gradient descent to calculate b1 and b2

def gradient_descent(x, y, b1, b2, learning_rate, iterations):
    for i in range(iterations):
        y_pred = b1 * x + b2
        b1 = b1 - (2/n) * learning_rate * (np.sum((y_pred - y) * x))
        b2 = b2 - (2/n) * learning_rate * (np.sum(y_pred - y))
    return b1, b2

b1, b2 = gradient_descent(x, y, 0, 0, 0.01, 1000)






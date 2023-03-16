# Linear Regression 

# example of a linear regression model

# example 1 - simple linear regression

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

# using the formula to calculate b1 and b2

def linear_reg(x, y):
    b1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    b2 = y_mean - (b1 * x_mean)
    return b1, b2

b1, b2 = linear_reg(x, y)

# print coefficients

print("b1 = ", b1)
print("b2 = ", b2)

# plot regression line

y_pred = b1 * x + b2

plt.scatter(x, y)
plt.plot(x, y_pred, color = "red")
plt.show()



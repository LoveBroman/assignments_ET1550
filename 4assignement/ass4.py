import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1-np.exp(-x))

sig = np.vectorize(sigmoid)

# Fråga 2

x = np.array([0.6, 0.8]).T

w1 = np.array([
    [0.2, 0.4,-0.1],
    [0.3, -0.5, 0.2]])

b1 = np.array([-0.1, 0.2, 0.3])

w2 = np.array([-0.4, 0.1, 0.6])

b2 = 0.2

z1 = sig(np.matmul(w1.T,x) + b1).T


print(np.matmul(w2.T,z1) + b2)

#Fråga 3
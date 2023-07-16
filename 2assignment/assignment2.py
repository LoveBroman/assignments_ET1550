import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore

## Uppgift 1
print("Fråga 1")

def normal_eq(A, b):
    tA = np.transpose(A)
    return np.matmul(np.linalg.inv(np.matmul(tA, A)), np.matmul(tA, b))

A = np.transpose(np.array(
    [[1 for _ in range(7)],
    [(i + 2 ) for i in range(7)]]))

b = np.transpose(np.array([68, 75, 83, 89, 92, 95, 98]))

x = normal_eq(A, b)

print(x)
x2 = np.linspace(0, 10, 1000)
y2 = x2 * x[1] + x[0]

# plt.plot(np.transpose(A)[1], b, "o")
# plt.plot(x2, y2)
# plt.show()

## Uppgift 2
print("Fråga 2")
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])
a = 0.01
def c_prim_w0(w0, w1):
    return -a *1/5 *sum([(y[i]- w0 - w1 * x[i]) for i in range(len(x))])

def c_prim_w1(w0, w1):
    return -a *1/5 * sum([x[i]*(y[i]- w0 - w1 * x[i]) for i in range(len(x))])

w0 = 0
w1 = 0
for i in range(2):
    w_temp = w0
    w0 -= c_prim_w0(w0, w1)
    w1 -= c_prim_w1(w_temp, w1)

print("parametrarnas värde efter 2 iterationer",w0,w1)

## 3 Jag ska utföra z-score normalisering på datasetet.
print("Fråga 3")
def normalize(dat):
    df = dat.copy()
    for col in df:
        df[col] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
    return df
data = pd.DataFrame({"age":[28, 35, 42, 25, 30],
                    "income":[60000, 70000, 80000, 55000, 65000]})

print(f"normaliserad data: \n {normalize(data)}")
## 4
print("fråga 4")

def mse(A, x, y):
   return 1/(2*len(x)) * sum((np.matmul(A, x) - y) ** 2)

def loocv(data, y):
    ms = 0
    for i in range(len(data)):
        t_data = np.delete(data, i, axis=0)
        t_y = np.delete(y, i)
        t_x = normal_eq(t_data, t_y)
        err = (t_x @ data[i] - y[i])**2
        ms += err
    return ms / len(data)

data1 = np.array([np.ones(5),[25, 30, 30, 20, 28], [50, 60, 70, 40, 55]]).T
y = np.array([200, 250, 300, 180, 220])

# synfeat = humidity * temprature
# If the dataset would be larger i would use cross validation here but due to the small
# I will look at the MSE on the primary data and compare them

synfeat = (data1.T[1] * data1.T[2]).reshape((-1, 1))
data2 = np.hstack((data1, synfeat))

x1 = normal_eq(data1, y)
ms1 = mse(data1, x1, y)
print(f"MSE utan syntfeat {ms1}")

x2 = normal_eq(data2, y)
ms2 = mse(data2,x2, y)
print(f"MSE med syntfeat {ms2}")

print("loocv utan synfeat ", loocv(data1, y))
print("loocv med synfeat ", loocv(data2, y))



# train_data1 = np.array([np.ones(3),[25, 30, 30], [50, 60, 70]]).T
# y_train = np.array([200, 250, 300])
# train_synfeat = (train_data1.T[1] * train_data1.T[2]).reshape((-1, 1))
# train_data2 = np.hstack((train_data1, train_synfeat))
# test_data1 = np.array([np.ones(2), [20, 28], [40, 55]]).T
# print(test_data1.T[1])
#
# test_synfeat = (test_data1.T[1] * test_data1.T[2]).reshape((-1, 1))
# test_data2 = np.hstack((test_data1, test_synfeat))
# y_test = np.array([180, 220])
#
# x11 = normal_eq(train_data1, y_train)
# x22 = normal_eq(train_data2, y_train)
# print(mse(test_data1, x11, y_test))
# print(mse(test_data2, x22, y_test))
# loocv(data1, y)
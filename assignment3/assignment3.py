import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Fråga 1
df = pd.DataFrame({
    "Discount" : [1, 0, 1, 0, 1, 1, 0, 0, 1, 1],
    "Urgent"   : [1, 0, 0, 1, 0, 1, 1, 1, 1, 0],
    "Free"     : [0, 1, 1, 0, 0, 0, 1, 0, 1, 1],
    "Spam"     : [1, 0, 1, 0, 0, 1, 1, 1, 1, 0]})

#print(df.head(10))
indata = [1, 0, 0]

def helper(indata, data, label, num):
    val = 1
    filtered_data = data[data[label] == num]
    for i, col in enumerate(data.columns[:-1]):
        p_a = len(filtered_data) / len(df)
        if indata[i]:
            p_b_a = filtered_data[col].sum() / len(filtered_data)
            p_b = data[col].sum() / len(data)
        else:
            p_b_a = (len(filtered_data) - filtered_data[col].sum()) / len(filtered_data)
            p_b = (len(data) - data[col].sum()) / len(data)
        p_a_b = (p_b_a * p_a) / p_b
        val *= p_a_b
    return val


# filtered_data_1 = df[df["Spam"] == 1]
# filtered_data_0 = df[df["Spam"] == 0]
#
#
# p_d1 = df["Discount"].sum() / len(df)
# p_d1_y = filtered_data_1["Discount"].sum()/len(filtered_data_1)
# py = filtered_data_1.sum()/len(df)
#
# p_y_d1 = (p_d1_y * py) / p_d1
#
# print(p_y_d1)
#
#
# sum_zero = filtered_data_0.sum()
# sum_one = filtered_data_1.sum()
# print(type(filtered_data_1))
#
# am_sum_zero = sum_zero


# print(sum_zero)
# print(sum_one)
print(helper(indata, df, "Spam", 0))
print(helper(indata, df, "Spam", 1))

#Fråga 2
b_data = pd.DataFrame({
    "Age" : [25, 30, 35, 40, 45],
    "Blood Pressure" : [120, 110, 130, 140, 115],
    "Condition" : [0, 1, 0, 1, 0]
})

def knn(df, indata,label, k=3):
    data = df.copy()
    x_data = data.drop(label, axis=1)
    data["dist"] = x_data.apply(lambda x: sum([(x[i] - indata[i])**2 for i in range(len(x))]), axis = 1)
    data_sorted = data.sort_values("dist")
    return data_sorted.head(k)[label].mode()[0]




classo = knn(b_data, [32, 125], "Condition")
print(classo)

#Fråga 3
m1 = lambda x: 2.14 + 1.22*x
m2 = lambda x: 96.75 - 41.48 * x + 6.67 * x**2 - 0.43 * x**3 + 0.01 * x**4

ms1 = (m1(12) - 17) ** 2
ms2 = (m2(12) - 17) ** 2
print(ms1, ms2)

#Modellen är overfittad då den har låg error rate på training set men desto högre på testset
#Detta innebär att modellen anpassas efter noise i datan,

#Fråga 3

import numpy as np
import pandas as pd
import os

# 使用绝对路径读取文件
dataPath = os.path.dirname(os.path.abspath(__file__)) + '\\data.txt'
data = pd.read_csv(dataPath, header=None, names=['Population', 'Profit'])
data.insert(0, 'One', 1)
print(data.head())

X = data.iloc[:, 0:2]
y = data.iloc[:, 2:3]
# X = np.array(X)  # 97 * 2
# y = np.array(y)  # 97 * 1
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix([0, 0])
print(np.sum(np.power(y, 2)))
# #
# x = np.array([[1, 2], [1, 3], [1, 4]])
# print(np.sum(x))
# y = np.array([1, 1, 1])
# theta = np.array([1, 1])
# print(x @ theta.T)
#
# re = (x @ theta.T) - y
# print('re:', re)
# re = np.power(re, 2)
# print(re)
# print(np.sum(re))

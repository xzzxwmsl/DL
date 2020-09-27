import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# plot 画图
# data_1 = pd.Series(np.random.randn(1000), index=np.arange(1000))
# data_1.cumsum()
# data_1.plot()
# plt.show()

fig, ax = plt.subplots()

# 在指定范围内采点
# 在0~100内均匀采点50个
x = np.linspace(start=0, stop=10, num=50)
f_x = 10 + x**2
ax.plot(x, f_x, 'r', label='TestFUnction')
# 以上绘制了一个 10+x^2的函数

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Test')
plt.show()
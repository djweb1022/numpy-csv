# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/5/17 15:04'



import math
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-10,10, 500)
a=np.array(x)

y1= 1. / (1 + np.exp(-x))
y2=math.e**(-x)/((1+math.e**(-x))**2)

plt.xlim(-11,11)
ax = plt.gca()# get current axis 获得坐标轴对象
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')# 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
plt.plot(x,y1,label='Sigmoid',linestyle="-", color="blue")
#plt.legend(loc=0,ncol=2)
plt.legend(['Sigmoid'])

plt.show()
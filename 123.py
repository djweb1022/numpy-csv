# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/5/8 21:42'

import numpy as np
import math

# aaa = 2.567
# bbb = '%.2f' % aaa
# ccc = math.ceil(aaa)
#
# print(bbb)
#
# print(ccc)

list_a = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

list_n = np.array(list_a)


np.random.shuffle(list_n)

test1 = list_n[1:]

test2 = list_n[:, 1:]

print()
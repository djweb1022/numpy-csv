# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/5/9 14:50'

import pandas as pd
import numpy as np

total_merge = pd.read_csv('../totalmerge/total_merge.csv')

kkk = total_merge['num_of_prev_attempts'].value_counts(dropna=False)

print(kkk)

# print(kkk_index)
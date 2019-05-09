# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/5/9 14:50'

import pandas as pd
import numpy as np

total_merge = pd.read_csv('../totalmerge/total_merge.csv')

kkk = total_merge['assessment_type'].value_counts(dropna=False)

# kkk_min = total_merge['studied_credits'].min()
#
# kkk_max = total_merge['studied_credits'].max()


print(kkk)
# print(kkk_min)
# print(kkk_max)

# print(kkk_index)
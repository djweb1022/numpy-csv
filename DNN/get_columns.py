# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/5/9 14:50'

import pandas as pd
import numpy as np

total_merge = pd.read_csv('../totalmerge/total_merge.csv')

kkk = total_merge['highest_education'].value_counts()

print(kkk)
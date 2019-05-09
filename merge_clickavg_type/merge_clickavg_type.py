# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/5/9 13:04'

import pandas as pd
import numpy as np

# merge_clickavg = pd.read_csv('../different_clickavg/merge_clickavg.csv')
#
# merge_type = pd.read_csv('../different_type/merge_type.csv')
#
# merge_clickavg.to_csv('merge_clickavg_index.csv')
#
# merge_type.to_csv('merge_type_index.csv')

merge_clickavg_index = pd.read_csv('merge_clickavg_index.csv')

merge_type_index = pd.read_csv('merge_type_index.csv')

print(merge_clickavg_index.columns)

merge_df = pd.merge(merge_clickavg_index, merge_type_index, how='outer', on='index')

merge_df.to_csv('clickavg_type.csv')

print()

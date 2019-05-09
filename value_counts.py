# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/5/8 19:27'

import pandas as pd
import numpy as np

# studentvle_vle = pd.read_csv('studentvle-vle.csv')
#
# kkk = studentvle_vle['activity_type'].value_counts()
#
# print(kkk)


rename_list = ['AAA_2013J', 'AAA_2014J', 'BBB_2013J', 'BBB_2014J', 'BBB_2013B', 'BBB_2014B', 'CCC_2014J', 'CCC_2014B',
               'DDD_2013J', 'DDD_2014J', 'DDD_2013B', 'DDD_2014B', 'EEE_2013J', 'EEE_2014J', 'EEE_2014B', 'FFF_2013J',
               'FFF_2014J', 'FFF_2013B', 'FFF_2014B', 'GGG_2013J', 'GGG_2014J', 'GGG_2014B']

for k in rename_list:

    studentvle_vle = pd.read_csv('different_userscore/%s_user-score.csv' % k)

    print(studentvle_vle)

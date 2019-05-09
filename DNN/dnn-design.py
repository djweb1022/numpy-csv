# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/5/9 13:57'

import pandas as pd
import numpy as np
from keras.utils import to_categorical

total_merge = pd.read_csv('../totalmerge/total_merge.csv')

"""
# 用户性别
# M(男性)=[1,0] 
# F(女性)=[0,1]
"""
list_gender = list(total_merge['gender'])
num_gender = []
# len_gender = len(list_gender)
# np_gender = np.zeros((len_gender, 2))

for gender in list_gender:
    if gender == 'M':
        num_gender.append(0)
    elif gender == 'F':
        num_gender.append(1)

cat_gender = to_categorical(num_gender, 2)

"""

用户地区

#0  East Anglian Region     18064
#1  Scotland                17733
#2  South Region            17398
#3  London Region           16156
#4  North Western Region    13723
#5  South West Region       13267
#6  West Midlands Region    12833
#7  East Midlands Region    12185
#8  South East Region       11638
#9  Wales                   11005
#10 Yorkshire Region        10197
#11 North Region            10103
#12 Ireland                  7201

# Name: region, dtype: int64

"""
list_region = list(total_merge['region'])
num_region = []

for region in list_region:
    if region == 'East Anglian Region':
        num_region.append(0)
    elif region == 'Scotland':
        num_region.append(1)
    elif region == 'South Region':
        num_region.append(2)
    elif region == 'London Region':
        num_region.append(3)
    elif region == 'North Western Region':
        num_region.append(4)
    elif region == 'South West Region':
        num_region.append(5)
    elif region == 'West Midlands Region':
        num_region.append(6)
    elif region == 'East Midlands Region':
        num_region.append(7)
    elif region == 'South East Region':
        num_region.append(8)
    elif region == 'Wales':
        num_region.append(9)
    elif region == 'Yorkshire Region':
        num_region.append(10)
    elif region == 'North Region':
        num_region.append(11)
    elif region == 'Ireland':
        num_region.append(12)

cat_region = to_categorical(num_region, 13)


"""

进入课程时的最高教育水平

#0 A Level or Equivalent          78837
#1 Lower Than A Level             62269
#2 HE Qualification               27113
#3 Post Graduate Qualification     1910
#4 No Formal quals                 1374

# Name: highest_education, dtype: int64

"""
list_highest_education = list(total_merge['highest_education'])
num_highest_education = []

for education in list_highest_education:
    if education == 'A Level or Equivalent':
        num_highest_education.append(0)
    elif education == 'Lower Than A Level':
        num_highest_education.append(1)
    elif education == 'HE Qualification':
        num_highest_education.append(2)
    elif education == 'Post Graduate Qualification':
        num_highest_education.append(3)
    elif education == 'No Formal quals':
        num_highest_education.append(4)

cat_highest_education = to_categorical(num_highest_education, 5)


print()















# 171503
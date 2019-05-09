# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/5/9 13:57'

import pandas as pd
import math
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


"""

剥夺指数,用于衡量一个地区的贫困程度，涉及7个领域：
收入、就业、健康剥夺与残疾、教育与职业技能培训剥夺、住房与服务间的障碍隔阂、居住环境剥夺、犯罪

#0  0-10%      14954
#1  10-20      16508
#2  20-30%     17271
#3  30-40%     18981
#4  40-50%     16841
#5  50-60%     16755
#6  60-70%     15939
#7  70-80%     16077
#8  80-90%     15616
#9  90-100%    14947
#10 NaN         7614(转为全0向量表示)

实际只有10类

# Name: imd_band, dtype: int64

"""
list_imd_band = list(total_merge['imd_band'])
num_imd_band = []

for imd in list_imd_band:
    if imd == '0-10%':
        num_imd_band.append(0)
    elif imd == '10-20':
        num_imd_band.append(1)
    elif imd == '20-30%':
        num_imd_band.append(2)
    elif imd == '30-40%':
        num_imd_band.append(3)
    elif imd == '40-50%':
        num_imd_band.append(4)
    elif imd == '50-60%':
        num_imd_band.append(5)
    elif imd == '60-70%':
        num_imd_band.append(6)
    elif imd == '70-80%':
        num_imd_band.append(7)
    elif imd == '80-90%':
        num_imd_band.append(8)
    elif imd == '90-100%':
        num_imd_band.append(9)
    elif math.isnan(imd):
        num_imd_band.append(10)

cat_imd_band = to_categorical(num_imd_band, 11)

# 删除最后一列
cat_imd_band = np.delete(cat_imd_band, -1, axis=1)


"""

年龄段

#0 0-35     117818
#1 35-55     52551
#2 55<=       1134

# Name: age_band, dtype: int64

"""
list_age_band = list(total_merge['age_band'])
num_age_band = []

for age in list_age_band:
    if age == '0-35':
        num_age_band.append(0)
    elif age == '35-55':
        num_age_band.append(1)
    elif age == '55<=':
        num_age_band.append(2)

cat_age_band = to_categorical(num_age_band, 3)


"""

学生尝试此课程的次数

#0 0    153296
#1 1     14712
#2 2      2767
#3 3       523
#4 4       153
#5 5        39
#6 6        13

# Name: num_of_prev_attempts, dtype: int64

"""
list_num_of_prev_attempts = list(total_merge['num_of_prev_attempts'])
num_num_of_prev_attempts = []

cat_num_of_prev_attempts = to_categorical(list_num_of_prev_attempts, 7)



print()















# 171503
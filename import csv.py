# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/5/7 9:43'

import os
import numpy as np
import csv

import pandas as pd

# a_1 = np.loadtxt('open-university-analytics-datasets-unzipped/courses.csv', dtype=str, usecols=(1,), delimiter=",", skiprows=0)
#
# print(a_1.shape)
# print(a_1)
#
# print(a_1[2])
#
# a = a_1[2]
# b = str(2014J)
#
# print(a == b)

df = pd.read_csv('open-university-analytics-datasets-unzipped/courses.csv')

print(df['code_presentation'].unique())

# print(df.loc[0]['module_presentation_length'] > 267)

print(df[df['module_presentation_length'] == 241].index)

list_df = list(df[df['module_presentation_length'] == 241].index)

print(list_df)

for num in list_df:
    print(df.loc[num, 'module_presentation_length'])
    df.loc[num, 'module_presentation_length'] += 1000

print(df)

df2 = df.copy()
df2['new_rw'] = '22'

print(df2[df2['module_presentation_length'] == 1241])

df3 = df2[df2['module_presentation_length'] == 1241]

# df.drop()

print(df3)

# npnp = np.array(df.values)
# print(npnp)
# print(npnp.shape)

# df.drop()

# 输出列信息
# print(df.columns)

# 输出
df3.to_csv('kakak.csv')


# print(df)

# print(df[df.module_presentation_length >= 250])
#
# print(df['code_module'][1] == 'AAA')
# print(len(df))
#
# for i in range(len(df)):
#     print(df[i:i+1])


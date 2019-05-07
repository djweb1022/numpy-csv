# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/5/7 23:15'


import pandas as pd
import numpy as np

studentvle = pd.read_csv('open-university-analytics-datasets-unzipped/studentVle.csv')

vle = pd.read_csv('open-university-analytics-datasets-unzipped/vle.csv')

df = pd.merge(studentvle, vle, how='left', on='id_site')

df.to_csv('studentvle-vle.csv', index=False)

print(df['activity_type'].value_counts())

# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/5/9 13:42'

import pandas as pd
import numpy as np

clickavg_type = pd.read_csv('../merge_clickavg_type/clickavg_type.csv')

student_info = pd.read_csv('../open-university-analytics-datasets-unzipped/studentInfo.csv')

total_merge = pd.merge(clickavg_type, student_info, how='left', on=['id_student', 'code_module', 'code_presentation'])

total_merge.to_csv('total_merge.csv')

print()
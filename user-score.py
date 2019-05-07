# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/5/7 17:09'

import pandas as pd
import numpy as np

df = pd.read_csv('open-university-analytics-datasets-unzipped/studentAssessment.csv')
# print(df)

# 筛选出is_banked=0的记录
# df2 = df[df['is_banked'].isin([0])]
df2 = df[df['is_banked'] == 0]
df2 = df2[df2['score'] >= 1]
df2.reset_index(inplace=True)
# print(df2.shape[0])

# 加入空值列'assessment_type'
# a = np.empty(df2.shape[0])
# a.fill(np.nan)
# df2.insert(df2.shape[1], 'assessment_type', a)
print(df2)


# df2['assessment_type'] = np.nan
# print(df2['id_assessment'].value_counts())
# print(df2.columns)
# df2.to_csv('ttt.csv')

# 以id_assessment为外键，df2为行基准合并studentAssessment.csv和assessments.csv
assessments = pd.read_csv('open-university-analytics-datasets-unzipped/assessments.csv')
df3 = pd.merge(df2, assessments, how='left', on='id_assessment')

print(df3)
print(df3.columns)

# df3.to_csv('user-score.csv', index=False)

# print(df3['assessment_type_y'].value_counts())
# print(assessments)
# print(assessments[assessments['id_assessment'] == 1752].index)
# print(df2.columns)
# df2_index = list(df2.columns).index('id_assessment')
# print(df2.iloc[5, 0])

# df3.iloc[5, 0] = df2.iloc[5, 1]
# print(df2.loc[0, 'id_assessment'])
# print(assessments.loc[0, 'id_assessment'])
#
# print(df2.loc[0, 'id_assessment'] == assessments.loc[0, 'id_assessment'])
#
# print(df2.loc[0, 'assessment_type'])
# print(assessments.loc[0, 'assessment_type'])
#
# df3 = df2.copy()
#
# df3.loc[0, 'assessment_type'] = assessments.loc[0, 'assessment_type']
# kkk = str(assessments.loc[0, 'assessment_type'])
# print(kkk)
# df2.loc[0, 'assessment_type'] = kkk

# for i in range(df2.shape[0]):
#     for j in range(assessments.shape[0]):
#         if df3.loc[i, 'id_assessment'] == assessments.loc[j, 'id_assessment']:
#             df3.loc[i, 'assessment_type'] = assessments.loc[j, 'assessment_type']
# #

# print(df2)


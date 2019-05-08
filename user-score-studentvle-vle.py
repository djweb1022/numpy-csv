# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/5/8 9:40'

import pandas as pd
import numpy as np

user_score = pd.read_csv('user-score.csv')
studentvle_vle = pd.read_csv('studentvle-vle.csv')

# 将出现过的id_student做个列表
list_unique_id_student = np.array(user_score[user_score['id_student'] > 0]['id_student'].unique())
print(len(list_unique_id_student))
print(list_unique_id_student)

for user_id in list_unique_id_student:

    # 对于某一用户的ID，分别找到两张表中的索引集合
    user_score_index = np.array(user_score[user_score['id_student'] == user_id].index)
    studentvle_vle_index = np.array(studentvle_vle[studentvle_vle['id_student'] == user_id].index)

    # 对studentvle_vle，生成一张筛选过user_id的子DataFrame
    user_id_df = studentvle_vle[studentvle_vle['id_student'] == user_id]
    print(user_id_df)

    # 用列表记录评分对应的日期
    list_user_score_time = []
    for i in user_score_index:
        list_user_score_time.append(user_score.loc[i, 'date'])

    # 将索引及对应日期装入DataFrame中，确保按日期进行升序排序
    timedict = {'date': list_user_score_time}
    time_df = pd.DataFrame(timedict, user_score_index)
    time_df_ascend = time_df.sort_values(by='date', ascending=True)
    time_df_ascend_index = list(time_df_ascend.index)
    time_df_ascend_date = list(time_df_ascend['date'])

    # 定义列表记录该用户分时间段的点击数之和
    list_clicksum = []

    # 对于user-score的每条记录
    for i in range(len(time_df_ascend_date)):
        if i == 0:
            before_date = -10000
            after_date = time_df_ascend_date[i]
        else:
            before_date = time_df_ascend_date[i-1]
            after_date = time_df_ascend_date[i]

        # 按user-score中记录日期，分段取出studentvle_vle中索引集合
        list_user_id_df_index = list(user_id_df[(before_date < user_id_df['date']) & (user_id_df['date'] <= after_date)].index)

        # 统计该时间区间内的点击总数，装入列表list_clicksum中
        clicksum = 0
        for j in list_user_id_df_index:
            clicksum += studentvle_vle.loc[j, 'sum_click']
        list_clicksum.append(clicksum)
        print(list_clicksum)





# list_index = list(user_score[user_score['id_student'] == 28400].index)
# print(list_index)
#
# list_index2 = list(studentvle_vle[studentvle_vle['id_student'] == 28400].index)
# print(list_index2)
#
# print(user_score.loc[501, 'date'] < studentvle_vle.loc[3909, 'date'])
# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/5/8 9:40'

import pandas as pd
import numpy as np

rename_list = ['AAA_2013J', 'AAA_2014J', 'BBB_2013J', 'BBB_2014J', 'BBB_2013B', 'BBB_2014B', 'CCC_2014J', 'CCC_2014B',
               'DDD_2013J', 'DDD_2014J', 'DDD_2013B', 'DDD_2014B', 'EEE_2013J', 'EEE_2014J', 'EEE_2014B', 'FFF_2013J',
               'FFF_2014J', 'FFF_2013B', 'FFF_2014B', 'GGG_2013J', 'GGG_2014J', 'GGG_2014B']

# 记录总共处理的数据条数和表数
sum_count = 0
table_count = 0

# 声明两张空的DataFrame，一张用于保存前一次处理完毕的表，一张用于保存合并后的总表
former_df = pd.DataFrame()
merge_df = pd.DataFrame()

for rename in rename_list:
    user_score = pd.read_csv('different_userscore/%s_user-score.csv' % rename)
    studentvle_vle = pd.read_csv('different_courses/%s_studentvle-vle.csv' % rename)
    studentregistration = pd.read_csv('open-university-analytics-datasets-unzipped/studentRegistration.csv')

    # 分离module和presentation
    rename_module = rename[:3]
    rename_presentation = rename[4:]

    # print(user_score)

    # 将出现过的id_student做个列表
    list_unique_id_student = np.array(user_score[user_score['id_student'] > 0]['id_student'].unique())
    # print(len(list_unique_id_student))
    # print(list_unique_id_student)

    # 循环计数
    number_record = 0

    for user_id in list_unique_id_student:

        # 对于某一用户的ID，分别找到两张表中的索引集合
        user_score_index = np.array(user_score[user_score['id_student'] == user_id].index)
        studentvle_vle_index = np.array(studentvle_vle[studentvle_vle['id_student'] == user_id].index)

        # 对于某一用户的ID，在studentRegistration.csv找到索引，查询该rename中初次注册日期
        studentregistration_index = list(studentregistration[(studentregistration['id_student'] == user_id) &
                                                             (studentregistration['code_module'] == rename_module) &
                                                             (studentregistration['code_presentation'] == rename_presentation)].index)
        first_registration = studentregistration.loc[studentregistration_index[0], 'date_registration']

        # 对studentvle_vle，生成一张筛选过user_id的子DataFrame
        user_id_df = studentvle_vle[studentvle_vle['id_student'] == user_id]
        # print(user_id_df)

        # 用列表记录评分对应的日期
        list_user_score_time = []
        for i in user_score_index:
            list_user_score_time.append(user_score.loc[i, 'date'])

        # 将索引及对应日期装入DataFrame中，确保按日期进行升序排序
        timedict = {'date': list_user_score_time}
        time_df = pd.DataFrame(timedict, user_score_index)
        time_df_ascend = time_df.sort_values(by='date', ascending=True)

        # 对某一用户的ID，生成索引列表和日期列表
        time_df_ascend_index = list(time_df_ascend.index)
        time_df_ascend_date = list(time_df_ascend['date'])

        # 定义列表记录该用户分时间段的点击数之和
        list_clicksum = []

        # 对于user-score的每条记录，取出时间区间，第一次需要获得注册时间
        for i in range(len(time_df_ascend_date)):
            if i == 0:
                before_date = first_registration
                after_date = time_df_ascend_date[i]
            else:
                before_date = time_df_ascend_date[i-1]
                after_date = time_df_ascend_date[i]

            # before_date = np.nan
            # after_date = np.nan

            # 计算时间区间
            dateplus = after_date-before_date+1

            # 按user-score中记录日期，分段取出studentvle_vle中索引集合
            list_user_id_df_index = list(user_id_df[(before_date < user_id_df['date']) & (user_id_df['date'] <= after_date)].index)

            # 统计该时间区间内的点击总数，日均点击数，装入列表list_clicksum中
            clicksum = 0
            clickavg = 0
            for j in list_user_id_df_index:
                clicksum += studentvle_vle.loc[j, 'sum_click']

            # 计算日均点击数，保留两位小数
            if dateplus <= 0:
                clickavg = np.nan
            else:
                f = clicksum / dateplus
                clickavg = '%.2f' % f

            # 向DataFrame中写入clickavg
            user_score.loc[time_df_ascend_index[i], 'clickavg'] = clickavg

            # 计数
            number_record += 1
            # print('已处理%s条记录' % str(number_record))

    # 保存数据表
    user_score.to_csv('different_clickavg/%s_clickavg.csv' % rename)
    print('已完成表%s,这张表共处理%s条记录' % (rename, str(number_record)))

    sum_count += number_record
    table_count += 1

    # 合并所有表
    if table_count == 1:
        former_df = user_score
    else:
        merge_df = pd.merge(former_df, user_score, how='outer')
        former_df = merge_df

merge_df.to_csv('different_clickavg/merge_clickavg.csv')
print('工作结束，共处理%s张表,共处理%s条记录' % (table_count, sum_count))


# 已处理171503条记录


# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/5/9 13:57'

import pandas as pd
import math
import numpy as np
from keras.utils import to_categorical

total_merge = pd.read_csv('../totalmerge/total_merge.csv')

"""

# 用户性别 gender
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

用户地区 region

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

进入课程时的最高教育水平 highest_education

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

剥夺指数 imd_band,用于衡量一个地区的贫困程度，涉及7个领域：
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

年龄段 age_band

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

学生尝试此课程的次数 num_of_prev_attempts

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

"""

学习信用 studied_credits
取值范围 30-630

#0 30-59
#1 60-89
#2 90-119
#3 120-149
#4 150-179
#5 >=180

# Name: studied_credits, dtype: int64

"""
list_studied_credits = list(total_merge['studied_credits'])
num_studied_credits = []

for cred_num in list_studied_credits:
    if (cred_num >= 30) & (cred_num <= 59):
        num_studied_credits.append(0)
    elif (cred_num >= 60) & (cred_num <= 89):
        num_studied_credits.append(1)
    elif (cred_num >= 90) & (cred_num <= 119):
        num_studied_credits.append(2)
    elif (cred_num >= 120) & (cred_num <= 149):
        num_studied_credits.append(3)
    elif (cred_num >= 150) & (cred_num <= 179):
        num_studied_credits.append(4)
    elif cred_num >= 180:
        num_studied_credits.append(5)

cat_studied_credits = to_categorical(num_studied_credits, 6)

"""

是否残疾 disability

#1 N    156708
#2 Y     14795

# Name: disability, dtype: int64

"""
list_disability = list(total_merge['disability'])
num_disability = []

for dis in list_disability:
    if dis == 'N':
        num_disability.append(0)
    elif dis == 'Y':
        num_disability.append(1)

cat_disability = to_categorical(num_disability, 2)

"""

期末测试结果 final_result

#0 Pass           105345
#1 Fail            27414
#2 Distinction     26238
#3 Withdrawn       12506

1通过 0不通过

# Name: final_result, dtype: int64

"""
list_final_result = list(total_merge['final_result'])
num_final_result = []

for result in list_final_result:
    if result == 'Pass':
        num_final_result.append(1)
    elif result == 'Fail':
        num_final_result.append(0)
    elif result == 'Distinction':
        num_final_result.append(0)
    elif result == 'Withdrawn':
        num_final_result.append(0)

# cat_final_result = to_categorical(num_final_result, 4)
cat_final_result = np.array(num_final_result)
# cat_final_result = np.reshape((cat_final_result.shape[0], 1))

"""

测试类型 assessment_type

#0 TMA     96732
#1 CMA     69815
#2 Exam     4956

# Name: assessment_type, dtype: int64

"""
list_assessment_type = list(total_merge['assessment_type'])
num_assessment_type = []

for astype in list_assessment_type:
    if astype == 'TMA':
        num_assessment_type.append(0)
    elif astype == 'CMA':
        num_assessment_type.append(1)
    elif astype == 'Exam':
        num_assessment_type.append(2)

cat_assessment_type = to_categorical(num_assessment_type, 3)

"""

测试权重 weight

0-1标准化

"""
list_weight = np.array(total_merge['weight'])
cat_weight = []
weight_min = list_weight.min()
weight_max = list_weight.max()
max_min = weight_max - weight_min

for weight in list_weight:
    ff = (weight - weight_min) / max_min
    num_weight = '%.3f' % ff
    cat_weight.append(num_weight)

cat_weight = np.array(cat_weight)

"""

日均点击数 clickavg

0-1标准化

取均值两倍作最大值，大于最大值的视为1

"""
list_clickavg = list(total_merge['clickavg'])
cat_clickavg = []
clickavg_min = 0
clickavg_mean = total_merge['clickavg'].mean()
clickavg_max = math.ceil(clickavg_mean*2)

for avg in list_clickavg:
    if math.isnan(avg):
        num_avg = 0
    elif avg > clickavg_max:
        avg = clickavg_max
        ff = (avg - clickavg_min)/(clickavg_max - clickavg_min)
        num_avg = '%.3f' % ff
    else:
        ff = (avg - clickavg_min)/(clickavg_max - clickavg_min)
        num_avg = '%.3f' % ff
    cat_clickavg.append(num_avg)

# df_clickavg = pd.DataFrame(cat_clickavg, columns=['avgavg'])
# df_clickavg = df_clickavg.fillna(0, method=None)

cat_clickavg = np.array(cat_clickavg)

"""

分数 score

0-1标准化

范围1-100

"""
list_score = list(total_merge['score'])
num_score = []

for score in list_score:
    new_score = score/100
    num_score.append(new_score)

cat_score = np.array(num_score)

"""列出统计的12种活动类型"""
list_activity_type = ['forumng', 'oucontent', 'subpage', 'homepage', 'quiz', 'resource', 'url', 'ouwiki',
                      'oucollaborate', 'externalquiz', 'page', 'questionnaire', ]

# 初始化空数组用于存放12列标准化之后的数值
lscat_activity_type = []

for str_type in list_activity_type:

    # 取出列
    list_str_type = list(total_merge[str_type])

    # 计算该列平均值
    type_mean = total_merge[str_type].mean()

    # 以平均值的两倍作为最大值(向上取整)
    type_max = math.ceil(type_mean*2)

    # 取出该列最小值
    type_min = total_merge[str_type].min()

    # 初始化空数组用于存放该列归一化之后的值
    list_num = []

    for num in list_str_type:
        if num > type_max:
            num = type_max
        ff = (num - type_min) / (type_max - type_min)
        num_type = '%.3f' % ff
        list_num.append(num_type)

    lscat_activity_type.append(list_num)

# 转为ndarray类型
cat_activity_type = np.array(lscat_activity_type).T


"""

用户情境：
#1 cat_disability          (171503,2)[:, :2]
#2 cat_gender              (171503,2)[:, 2:4]
#3 cat_age_band            (171503,3)[:, 4:7]
#4 cat_highest_education   (171503,5)[:, 7:12]
#5 cat_num_of_prev_attempts(171503,7)[:, 12:19]
#6 cat_imd_band            (171503,10)[:, 19:29]
#7 cat_region              (171503,13)[:, 29:42]

交互情境：
#8 cat_clickavg            (171503,)
#9 cat_weight              (171503,)
#10 cat_studied_credits     (171503,6)
#11 cat_activity_type       (171503,12)

平台情境：
#12 cat_assessment_type     (171503,3)

标签：
#13 cat_score               (171503,)
#14 cat_final_result        (171503,)

"""

# 按情境分类将数组进行横向拼接
merge_1_7 = np.hstack([cat_disability, cat_gender, cat_age_band, cat_highest_education, cat_num_of_prev_attempts,
                       cat_imd_band, cat_region])

# 将一维数组调整为一列
reshape_8 = cat_clickavg.reshape((cat_clickavg.shape[0], 1))
reshape_9 = cat_weight.reshape((cat_weight.shape[0], 1))

merge_8_11 = np.hstack([reshape_8, reshape_9, cat_studied_credits, cat_activity_type])
# merge_8_11 = np.around(merge_8_11, decimals=3)

merge_12 = cat_assessment_type

reshape_13 = cat_score.reshape((cat_score.shape[0], 1))

reshape_14 = cat_final_result.reshape((cat_final_result.shape[0], 1))

merge_13_14 = np.hstack([reshape_13, reshape_14])

# 形成总二维数组
merge_1_14 = np.hstack([merge_1_7, merge_8_11, merge_12, merge_13_14])

# 将数组中所有元素转化为float32类型
merge_final = merge_1_14.astype('float32')

# 随机打乱样本顺序
np.random.shuffle(merge_final)


import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.layers import Embedding, Input, Dense, merge, Reshape, Flatten, Dropout
from keras import regularizers
from keras import initializers
from keras.regularizers import l1, l2, l1_l2
from keras.layers.merge import multiply, concatenate

"""
train_data_1 = merge_final[:150000, 7:12]
train_data_2 = merge_1_7[:150000, 7:12]
train_label_1 = merge_final[:150000, 66:]

val_data_1 = merge_final[150000:170000, 7:12]
val_data_2 = merge_1_7[150000:170000, 7:12]
val_label_1 = merge_final[150000:170000, 66:]


input1 = Input(shape=(train_data_1.shape[1],), name="input1")
input1_hid_1 = Dense(8, activation='softmax')(input1)
input1_hid_2 = Dense(4, activation='softmax')(input1_hid_1)
input1_hid_3 = Dense(2, activation='softmax')(input1_hid_2)
input1_hid_4 = Dense(1, activation='softmax')(input1_hid_3)

output1 = Dense(1, activation='sigmoid', name="output1")(input1_hid_3)

model = Model(inputs=[input1], outputs=[output1])

model.compile(loss={'output1': 'binary_crossentropy'},
              optimizer=Adam(lr=10, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
              metrics=['accuracy'])

history = model.fit(train_data_1,
                    train_label_1,
                    batch_size=4096,
                    epochs=200,
                    validation_data=([val_data_1], [val_label_1]))
"""


"""

模型设计A

为split_1、split_2、split_3分别设计神经网络，中间合并成一个大网络，最后有2个输出

"""



split_1 = merge_final[:, :42]

split_2 = merge_final[:, 42:62]

split_3 = merge_final[:, 62:65]

split_4 = merge_final[:, 65:]

# 训练数据集
train_data_1 = split_1[:150000]
train_data_2 = split_2[:150000]
train_data_3 = split_3[:150000]
# 训练答案集
train_label_1 = split_4[:150000, :1]
train_label_2 = split_4[:150000, 1:]

# 验证数据集
val_data_1 = split_1[150000:160000]
val_data_2 = split_2[150000:160000]
val_data_3 = split_3[150000:160000]
# 验证答案集
val_label_1 = split_4[150000:160000, :1]
val_label_2 = split_4[150000:160000, 1:]



# 定义三个输入，分别代表用户、交互、平台情境
user_input = Input(shape=(17,), name="user_input")
# item_input = Input(shape=(split_2.shape[1],), name="input2")
item_input = Input(shape=(1,), name="item_input")


def init_normal(shape, name=None):
    return initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)


# Embedding layer
MF_Embedding_User = Embedding(input_dim=2, output_dim=10, name='mf_embedding_user',
                              embeddings_regularizer=l2(0), input_length=17)(user_input)
MF_Embedding_Item = Embedding(input_dim=2, output_dim=10, name='mf_embedding_item',
                              embeddings_regularizer=l2(0), input_length=2)(item_input)

MLP_Embedding_User = Embedding(input_dim=2, output_dim=5, name="mlp_embedding_user",
                               embeddings_regularizer=l2(0), input_length=1)(user_input)
MLP_Embedding_Item = Embedding(input_dim=2, output_dim=5, name='mlp_embedding_item',
                               embeddings_regularizer=l2(0), input_length=1)(item_input)


# MF part
mf_user_latent = Flatten()(MF_Embedding_User)
mf_item_latent = Flatten()(MF_Embedding_Item)
mf_vector = multiply([mf_user_latent, mf_item_latent]) # element-wise multiply

# MLP part
mlp_user_latent = Flatten()(MLP_Embedding_User)
mlp_item_latent = Flatten()(MLP_Embedding_Item)
mlp_vector = concatenate([mlp_user_latent, mlp_item_latent])

layers = [16, 8, 4]

layer1 = Dense(16, kernel_regularizer=l2(0), activation='relu', name="layer1")(mlp_vector)
layer2 = Dense(8, kernel_regularizer=l2(0), activation='relu', name="layer2")(layer1)
layer3 = Dense(4, kernel_regularizer=l2(0), activation='relu', name="layer3")(layer2)


predict_vector = concatenate([mf_vector, layer3])

# Final prediction layer
prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)


model = Model(inputs=[user_input, item_input],
              outputs=[prediction])

model.summary()

# 为每个输入添加1个隐藏层
# hid_1 = Dense(128, activation='softmax')(input1)
# hid_1_2 = Dense(128, activation='softmax')(hid_1)
# hid_1_3 = Dense(128, activation='softmax')(hid_1_2)
#
# hid_2 = Dense(64, activation='softmax')(input2)
# hid_2_2 = Dense(64, activation='softmax')(hid_2)
# hid_2_3 = Dense(64, activation='softmax')(hid_2_2)
#
# hid_3 = Dense(32, activation='softmax')(input3)
# hid_3_2 = Dense(32, activation='softmax')(hid_3)
# hid_3_3 = Dense(32, activation='softmax')(hid_3_2)


# 合并3个隐藏层
# hid_all = keras.layers.concatenate([hid_1_3, hid_2_3, hid_3_3])


# 在合并的基础上再加3个隐藏层
# hid_all_1 = Dense(128, activation='softmax')(hid_all)
# hid_all_2 = Dense(64, activation='softmax')(hid_all_1)
# hid_all_3 = Dense(32, activation='softmax')(hid_all_2)

# hid_all_3_1_1 = Dense(32, activation='sigmoid')(hid_all_3)
# hid_all_3_1_2 = Dense(16, activation='sigmoid')(hid_all_3_1_1)
#
# hid_all_3_2_1 = Dense(32, activation='sigmoid')(hid_all_3)
# hid_all_3_2_2 = Dense(16, activation='sigmoid')(hid_all_3_2_1)

# 定义2个输出层，一个输出分数cat_score，一个输出期末测试结果cat_final_result
# output1 = Dense(1, activation='sigmoid', name="output1")(hid_all_3)
# output2 = Dense(4, activation='softmax', name="output2")(hid_all_3_2_2)


# 定义深度学习模型的输入和输出
# model = Model(inputs=[input1, input2, input3], outputs=[output1])
# model = Model(inputs=[input1, input2, input3], outputs=[output1, output2])

model.summary()

model.compile(loss={'output1': 'binary_crossentropy'},
              optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-06),
              metrics=['accuracy'])
# loss_weights = [1, 1],
# loss={'output1': 'mean_absolute_error', 'output2': 'categorical_crossentropy'},
# train_1 = split_1[:150000]
# train_2 = split_2[:150000]
# train_3 = split_3[:150000]
#
# label_1 = split_4[:150000, :1]
# label_2 = split_4[:150000, 2:]

history = model.fit({'input1': train_data_1, 'input2': train_data_2, 'input3': train_data_3},
                    {'output1': train_label_1},
                    batch_size=4096,
                    epochs=2000,
                    validation_data=([val_data_1, val_data_2, val_data_3], [val_label_1]))

history_dict = history.history
print(history_dict.keys())
print(history_dict)



# {'output1': train_label_1, 'output2': train_label_2},
# merge_final = np.around(merge_1_14_float32, decimals=3)

# split_1 = merge_1_14[:, :42]
#
# split_2 = merge_1_14[:, 42:62]
#
# split_3 = merge_1_14[:, 62:65]
#
# split_4 = merge_1_14[:, 65:]



# merge_13_14 = np.hstack([cat_score, cat_final_result])

# k_1 = cat_weight.shape
# k_2 = cat_clickavg.shape
#
# kkk = np.vstack([cat_weight, cat_clickavg]).T
#
# KKK_1 = kkk[1]



print()

# 171503

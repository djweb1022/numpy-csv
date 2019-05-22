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
# M(男性)=1 
# F(女性)=2

"""
list_gender = list(total_merge['gender'])
num_gender = []
# len_gender = len(list_gender)
# np_gender = np.zeros((len_gender, 2))

for gender in list_gender:
    if gender == 'M':
        num_gender.append(1)
    elif gender == 'F':
        num_gender.append(2)

cat_gender = np.array(num_gender)
cat_gender = cat_gender.reshape(cat_gender.shape[0], 1)

"""

用户地区 region

#1  East Anglian Region     18064
#2  Scotland                17733
#3  South Region            17398
#4  London Region           16156
#5  North Western Region    13723
#6  South West Region       13267
#7  West Midlands Region    12833
#8  East Midlands Region    12185
#9  South East Region       11638
#10 Wales                   11005
#11 Yorkshire Region        10197
#12 North Region            10103
#13 Ireland                  7201

# Name: region, dtype: int64

"""
list_region = list(total_merge['region'])
num_region = []

for region in list_region:
    if region == 'East Anglian Region':
        num_region.append(1)
    elif region == 'Scotland':
        num_region.append(2)
    elif region == 'South Region':
        num_region.append(3)
    elif region == 'London Region':
        num_region.append(4)
    elif region == 'North Western Region':
        num_region.append(5)
    elif region == 'South West Region':
        num_region.append(6)
    elif region == 'West Midlands Region':
        num_region.append(7)
    elif region == 'East Midlands Region':
        num_region.append(8)
    elif region == 'South East Region':
        num_region.append(9)
    elif region == 'Wales':
        num_region.append(10)
    elif region == 'Yorkshire Region':
        num_region.append(11)
    elif region == 'North Region':
        num_region.append(12)
    elif region == 'Ireland':
        num_region.append(13)

cat_region = np.array(num_region)
cat_region = cat_region.reshape(cat_region.shape[0], 1)

"""

进入课程时的最高教育水平 highest_education

#1 A Level or Equivalent          78837
#2 Lower Than A Level             62269
#3 HE Qualification               27113
#4 Post Graduate Qualification     1910
#5 No Formal quals                 1374

# Name: highest_education, dtype: int64

"""
list_highest_education = list(total_merge['highest_education'])
num_highest_education = []

for education in list_highest_education:
    if education == 'A Level or Equivalent':
        num_highest_education.append(1)
    elif education == 'Lower Than A Level':
        num_highest_education.append(2)
    elif education == 'HE Qualification':
        num_highest_education.append(3)
    elif education == 'Post Graduate Qualification':
        num_highest_education.append(4)
    elif education == 'No Formal quals':
        num_highest_education.append(5)

cat_highest_education = np.array(num_highest_education)
cat_highest_education = cat_highest_education.reshape(cat_highest_education.shape[0], 1)

"""

剥夺指数 imd_band,用于衡量一个地区的贫困程度，涉及7个领域：
收入、就业、健康剥夺与残疾、教育与职业技能培训剥夺、住房与服务间的障碍隔阂、居住环境剥夺、犯罪

#1  0-10%      14954
#2  10-20      16508
#3  20-30%     17271
#4  30-40%     18981
#5  40-50%     16841
#6  50-60%     16755
#7  60-70%     15939
#8  70-80%     16077
#9  80-90%     15616
#10  90-100%    14947
#11 NaN         7614

实际只有10类

# Name: imd_band, dtype: int64

"""
list_imd_band = list(total_merge['imd_band'])
num_imd_band = []

for imd in list_imd_band:
    if imd == '0-10%':
        num_imd_band.append(1)
    elif imd == '10-20':
        num_imd_band.append(2)
    elif imd == '20-30%':
        num_imd_band.append(3)
    elif imd == '30-40%':
        num_imd_band.append(4)
    elif imd == '40-50%':
        num_imd_band.append(5)
    elif imd == '50-60%':
        num_imd_band.append(6)
    elif imd == '60-70%':
        num_imd_band.append(7)
    elif imd == '70-80%':
        num_imd_band.append(8)
    elif imd == '80-90%':
        num_imd_band.append(9)
    elif imd == '90-100%':
        num_imd_band.append(10)
    elif math.isnan(imd):
        num_imd_band.append(11)

cat_imd_band = np.array(num_imd_band)
cat_imd_band = cat_imd_band.reshape(cat_imd_band.shape[0], 1)

"""

年龄段 age_band

#1 0-35     117818
#2 35-55     52551
#3 55<=       1134

# Name: age_band, dtype: int64

"""
list_age_band = list(total_merge['age_band'])
num_age_band = []

for age in list_age_band:
    if age == '0-35':
        num_age_band.append(1)
    elif age == '35-55':
        num_age_band.append(2)
    elif age == '55<=':
        num_age_band.append(3)

cat_age_band = np.array(num_age_band)
cat_age_band = cat_age_band.reshape(cat_age_band.shape[0], 1)

"""

学生尝试此课程的次数 num_of_prev_attempts

#1 0    153296
#2 1     14712
#3 2      2767
#4 3       523
#5 4       153
#6 5        39
#7 6        13

# Name: num_of_prev_attempts, dtype: int64

"""
list_num_of_prev_attempts = list(total_merge['num_of_prev_attempts'])
list_num_of_prev_attempts_1 = [i + 1 for i in list_num_of_prev_attempts]

cat_num_of_prev_attempts = np.array(list_num_of_prev_attempts_1)
cat_num_of_prev_attempts = cat_num_of_prev_attempts.reshape(cat_num_of_prev_attempts.shape[0], 1)

"""

学习信用 studied_credits
取值范围 30-630

#1 30-59
#2 60-89
#3 90-119
#4 120-149
#5 150-179
#6 >=180

# Name: studied_credits, dtype: int64

"""
list_studied_credits = list(total_merge['studied_credits'])
num_studied_credits = []

for cred_num in list_studied_credits:
    if (cred_num >= 30) & (cred_num <= 59):
        num_studied_credits.append(1)
    elif (cred_num >= 60) & (cred_num <= 89):
        num_studied_credits.append(2)
    elif (cred_num >= 90) & (cred_num <= 119):
        num_studied_credits.append(3)
    elif (cred_num >= 120) & (cred_num <= 149):
        num_studied_credits.append(4)
    elif (cred_num >= 150) & (cred_num <= 179):
        num_studied_credits.append(5)
    elif cred_num >= 180:
        num_studied_credits.append(6)

cat_studied_credits = np.array(num_studied_credits)
cat_studied_credits = cat_studied_credits.reshape(cat_studied_credits.shape[0], 1)

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
        num_disability.append(1)
    elif dis == 'Y':
        num_disability.append(2)

cat_disability = np.array(num_disability)
cat_disability = cat_disability.reshape(cat_disability.shape[0], 1)

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
cat_final_result = cat_final_result.reshape((cat_final_result.shape[0], 1))

"""

测试类型 assessment_type

#1 TMA     96732
#2 CMA     69815
#3 Exam     4956

# Name: assessment_type, dtype: int64

"""
list_assessment_type = list(total_merge['assessment_type'])
num_assessment_type = []

for astype in list_assessment_type:
    if astype == 'TMA':
        num_assessment_type.append(1)
    elif astype == 'CMA':
        num_assessment_type.append(2)
    elif astype == 'Exam':
        num_assessment_type.append(3)

cat_assessment_type = np.array(num_assessment_type)
cat_assessment_type = cat_assessment_type.reshape(cat_assessment_type.shape[0], 1)

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
clickavg_max = math.ceil(clickavg_mean * 2)

for avg in list_clickavg:
    if math.isnan(avg):
        num_avg = 0
    elif avg > clickavg_max:
        avg = clickavg_max
        ff = (avg - clickavg_min) / (clickavg_max - clickavg_min)
        num_avg = '%.3f' % ff
    else:
        ff = (avg - clickavg_min) / (clickavg_max - clickavg_min)
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
    if score >= 90:
        num_score.append(1)
    else:
        num_score.append(0)

cat_score = np.array(num_score)
cat_score = cat_score.reshape((cat_score.shape[0], 1))

"""
用户ID

"""
list_id_student = list(total_merge['id_student'])
list_id_student_set = list(set(list_id_student))

# 把ID转化为从1开始的连续值
list_id_student_con = []
for num_id in list_id_student:
    get_set = list_id_student_set.index(num_id)+1
    list_id_student_con.append(get_set)

list_id_student_con = np.array(list_id_student_con)
max_id_student = list_id_student_con.max()

cat_id_student = list_id_student_con.reshape((list_id_student_con.shape[0], 1))

# index_id_student = []
# element_id_student = []
#
# for i, element in enumerate(list_id_student):
#     if element not in element_id_student:
#         index_id_student.append(i)
#         element_id_student.append(element)
#
# test1 = np.array(index_id_student)
# test1 = test1.reshape((test1.shape[0]), 1)
# test2 = np.array(element_id_student)
# test2 = test2.reshape((test2.shape[0]), 1)
#
# test_all = np.hstack([test1, test2])
#
# cat_id_student = np.array(list_id_student)
# cat_id_student = cat_id_student.reshape(cat_id_student.shape[0], 1)

"""
测试ID

"""
list_id_assessment = list(total_merge['id_assessment'])
list_id_assessment_set = list(set(list_id_assessment))

# 把ID转化为从1开始的连续值
list_id_assessment_con = []
for num_id in list_id_assessment:
    get_set = list_id_assessment_set.index(num_id)+1
    list_id_assessment_con.append(get_set)

list_id_assessment_con = np.array(list_id_assessment_con)
max_id_assessment = list_id_assessment_con.max()

cat_id_assessment = list_id_assessment_con.reshape(list_id_assessment_con.shape[0], 1)

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
    type_max = math.ceil(type_mean * 2)

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
#1 cat_disability          (171503,1)
#2 cat_gender              (171503,1)
#3 cat_age_band            (171503,1)
#4 cat_highest_education   (171503,1)
#5 cat_num_of_prev_attempts(171503,1)
#6 cat_imd_band            (171503,1)
#7 cat_region              (171503,1)

交互情境：
#8 cat_clickavg            (171503,)
#9 cat_weight              (171503,)
#10 cat_studied_credits     (171503,1)
#11 cat_activity_type       (171503,12)

平台情境：
#12 cat_assessment_type     (171503,1)

标签：
#13 cat_score               (171503,)
#14 cat_final_result        (171503,1)

用户ID:
#15 cat_id_student          (171503,1)

测试ID：
#16 cat_id_assessment       (171503,1)

"""

merge_final = np.hstack([cat_id_student, cat_imd_band, cat_region, cat_id_assessment, cat_score])

# 将数组中所有元素转化为float32类型
# merge_final = merge_final.astype('float32')

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

模型设计A


"""
column_1 = 150000
column_2 = 160000

# 训练数据集
train_cat_id_student = merge_final[:column_1, :1]
train_cat_imd_band = merge_final[:column_1, 1:2]
train_cat_region = merge_final[:column_1, 2:3]
train_cat_id_assessment = merge_final[:column_1, 3:4]
train_label = merge_final[:column_1, 4:]

# 验证数据集
val_cat_id_student = merge_final[column_1:column_2, :1]
val_cat_imd_band = merge_final[column_1:column_2, 1:2]
val_cat_region = merge_final[column_1:column_2, 2:3]
val_cat_id_assessment = merge_final[column_1:column_2, 3:4]
val_label = merge_final[column_1:column_2, 4:]

# 测试集
test_cat_id_student = merge_final[column_2:, :1]
test_cat_imd_band = merge_final[column_2:, 1:2]
test_cat_region = merge_final[column_2:, 2:3]
test_cat_id_assessment = merge_final[column_2:, 3:4]
test_label = merge_final[column_2:, 4:]

# max_cat_id_student = merge_final[:, :1].max()
max_cat_imd_band = merge_final[:, 1:2].max()
max_cat_region = merge_final[:, 2:3].max()
# max_cat_id_assessment = merge_final[:, 3:4].max()

# 定义三个输入，分别代表用户、交互、平台情境
input_cat_id_student = Input(shape=(1,), name="input_cat_id_student")
input_cat_imd_band = Input(shape=(1,), name="input_cat_imd_band")
input_cat_region = Input(shape=(1,), name="input_cat_region")

input_cat_id_assessment = Input(shape=(1,), name="input_cat_id_assessment")


def init_normal(shape, name=None):
    return initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)


l2_regularizer = 0.001

# Embedding layer
MF_Embedding_cat_id_student = Embedding(input_dim=max_id_student + 1, output_dim=10,
                                        name='MF_Embedding_cat_id_student', embeddings_regularizer=l2(l2_regularizer),
                                        input_length=1)(input_cat_id_student)
MF_Embedding_cat_imd_band = Embedding(input_dim=max_cat_imd_band + 1, output_dim=10,
                                      name='MF_Embedding_cat_imd_band', embeddings_regularizer=l2(l2_regularizer),
                                      input_length=1)(input_cat_imd_band)
MF_Embedding_cat_region = Embedding(input_dim=max_cat_region + 1, output_dim=10,
                                    name='MF_Embedding_cat_region', embeddings_regularizer=l2(l2_regularizer),
                                    input_length=1)(input_cat_region)
MF_Embedding_cat_id_assessment = Embedding(input_dim=max_id_assessment + 1, output_dim=10,
                                           name='MF_Embedding_cat_id_assessment',
                                           embeddings_regularizer=l2(l2_regularizer),
                                           input_length=1)(input_cat_id_assessment)

MLP_Embedding_cat_id_student = Embedding(input_dim=max_id_student + 1, output_dim=5,
                                         name="MLP_Embedding_cat_id_student", embeddings_regularizer=l2(l2_regularizer),
                                         input_length=1)(input_cat_id_student)
# MLP_Embedding_cat_imd_band = Embedding(input_dim=max_cat_imd_band + 1, output_dim=5,
#                                        name="MLP_Embedding_cat_imd_band", embeddings_regularizer=l2(l2_regularizer),
#                                        input_length=1)(input_cat_imd_band)
# MLP_Embedding_cat_region = Embedding(input_dim=max_cat_region + 1, output_dim=5,
#                                      name="MLP_Embedding_cat_region", embeddings_regularizer=l2(l2_regularizer),
#                                      input_length=1)(input_cat_region)
MLP_Embedding_cat_id_assessment = Embedding(input_dim=max_id_assessment + 1, output_dim=5,
                                            name="MLP_Embedding_cat_id_assessment",
                                            embeddings_regularizer=l2(l2_regularizer),
                                            input_length=1)(input_cat_id_assessment)

# MF part
mf_cat_id_student = Flatten()(MF_Embedding_cat_id_student)
mf_cat_imd_band = Flatten()(MF_Embedding_cat_imd_band)
mf_cat_region = Flatten()(MF_Embedding_cat_region)
mf_cat_id_assessment = Flatten()(MF_Embedding_cat_id_assessment)


user_item_vector = multiply([mf_cat_id_student, mf_cat_id_assessment])
mf_vector = multiply([mf_cat_imd_band, mf_cat_region])  # element-wise multiply

# MLP part
mlp_cat_id_student = Flatten()(MLP_Embedding_cat_id_student)
# mlp_cat_imd_band = Flatten()(MLP_Embedding_cat_imd_band)
# mlp_cat_region = Flatten()(MLP_Embedding_cat_region)
mlp_cat_id_assessment = Flatten()(MLP_Embedding_cat_id_assessment)

mlp_vector = concatenate([mf_cat_id_student, mf_cat_id_assessment, mf_cat_imd_band, mf_cat_region])

layers = [16, 8, 4]

function_a = 'elu'

layer1 = Dense(32, kernel_regularizer=l2(0), activation=function_a, name="layer1")(mlp_vector)
layer2 = Dense(16, kernel_regularizer=l2(0), activation=function_a, name="layer2")(layer1)
layer3 = Dense(8, kernel_regularizer=l2(0), activation=function_a, name="layer3")(layer2)

predict_vector = concatenate([user_item_vector, mf_vector, layer3])

# Final prediction layer
prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)

model = Model(inputs=[input_cat_id_student, input_cat_id_assessment, input_cat_imd_band, input_cat_region],
              outputs=[prediction])

model.summary()

model.compile(loss={'prediction': 'binary_crossentropy'},
              optimizer=Adam(),
              metrics=['accuracy'])

history = model.fit([train_cat_id_student, train_cat_id_assessment, train_cat_imd_band, train_cat_region],
                    [train_label],
                    batch_size=4096,
                    epochs=200,
                    validation_data=([val_cat_id_student, val_cat_id_assessment, val_cat_imd_band, val_cat_region],
                                     [val_label]))

history_dict = history.history
print(history_dict.keys())
print(history_dict)

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# In[38]:


plt.clf()  # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

results = model.evaluate([test_cat_id_student, test_cat_id_assessment, test_cat_imd_band, test_cat_region],
                         [test_label])
print(results)

# 171503

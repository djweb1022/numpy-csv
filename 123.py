# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/5/8 21:42'

import keras
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras import regularizers

# aaa = 2.567
# bbb = '%.2f' % aaa
# ccc = math.ceil(aaa)
#
# print(bbb)
#
# print(ccc)

# list_a = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
#
# list_n = np.array(list_a)
#
#
# np.random.shuffle(list_n)
#
# test1 = list_n[1:]
#
# test2 = list_n[:, 1:]


rand_list = np.random.randint(1,100,[100000,1])

label_list = []

for i in rand_list:
    if i[0] > 50:
        label_list.append(1)
    else:
        label_list.append(0)

label_list = np.array(label_list)

label_list = label_list.reshape((label_list.shape[0], 1))


train_data_1 = rand_list[:80000]
train_label_1 = label_list[:80000]

val_data_1 = rand_list[80000:]
val_label_1 = label_list[80000:]

input1 = Input(shape=(train_data_1.shape[1],), name="input1")
input1_hid_1 = Dense(16, activation='relu')(input1)
input1_hid_2 = Dense(8, activation='relu')(input1_hid_1)
input1_hid_3 = Dense(4, activation='relu')(input1_hid_2)

output1 = Dense(1, activation='sigmoid', name="output1")(input1_hid_3)

model = Model(inputs=[input1], outputs=[output1])

model.compile(loss={'output1': 'binary_crossentropy'},
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
              metrics=['accuracy'])

history = model.fit(train_data_1,
                    train_label_1,
                    batch_size=1024,
                    epochs=200,
                    validation_data=([val_data_1], [val_label_1]))


print()
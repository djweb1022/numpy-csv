# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/5/22 23:07'

from matplotlib import pyplot as plt
import numpy as np
import mpl_toolkits.axisartist as axisartist
from matplotlib.ticker import MultipleLocator, FuncFormatter
import matplotlib.font_manager as fm
import pandas as pd
import math
import numpy as np

# 微软雅黑
my_font = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc', size=18)

only_gmf_2c = pd.read_csv('only_gmf_2c.csv')
only_dnn_2c = pd.read_csv('only_dnn_2c.csv')
deepca_2c = pd.read_csv('deepca_2c.csv')


def trainacc_2c():
    train_acc_1 = np.array(only_gmf_2c['val_f1_score'])
    train_acc_2 = np.array(only_dnn_2c['val_f1_score'])
    train_acc_3 = np.array(deepca_2c['val_f1_score'])

    epochs = range(1, len(train_acc_1)+1)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # ax.spines['left'].set_position(('data', 0))
    # ax.spines['bottom'].set_position(('data', 0))
    # plt.xlim([1, len(train_acc_1)])
    # plt.ylim([0, 100])
    # ax.set_xticks([0,1])
    # ax.set_yticks([-1, 2, 4, 6, 8, 10])

    ax.plot(epochs, train_acc_1, label="1", linestyle="--")
    ax.plot(epochs, train_acc_2, label="2", linestyle="-.")
    ax.plot(epochs, train_acc_3, label="3", linestyle="-")
    ax.tick_params(labelsize=18)

    plt.xlabel('迭代次数', fontproperties=my_font)
    plt.ylabel('F1-score', fontproperties=my_font)

    plt.legend(loc=4, prop=my_font)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    trainacc_2c()

print()

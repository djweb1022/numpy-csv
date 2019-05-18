# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/5/17 15:04'

from matplotlib import pyplot as plt
import numpy as np
import mpl_toolkits.axisartist as axisartist
from matplotlib.ticker import MultipleLocator, FuncFormatter


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    return np.where(x < 0, 0, x)


def prelu(x):
    return np.where(x < 0, 0.1 * x, x)


def elu(x):
    return np.where(x < 0, np.exp(x)-1, x)


def plot_sigmoid():
    x = np.linspace(-10.5, 10.5, 500)
    y = sigmoid(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax = axisartist.Subplot(fig, 111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    # ax.spines['left'].set_color('none')
    # ax.axis['bottom'].set_axisline_style("-|>", size=1.5)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.plot(x, y, label='sigmoid', linestyle="-", color="blue")
    plt.xlim([-10.5, 10.5])
    plt.ylim([-0.02, 1.02])
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([0.2,0.4, 0.6, 0.8, 1])
    ax.tick_params(labelsize=18)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    plt.tight_layout()
    plt.legend(loc=2, fontsize=18)
    plt.savefig("sigmoid.png")
    plt.show()


def plot_tanh():
    x = np.linspace(-10.5, 10.5, 500)
    y = tanh(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    # ax.spines['left'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.plot(x, y, label='tanh', linestyle="-", color="red")
    plt.xlim([-10.5, 10.5])
    plt.ylim([-1.02, 1.02])
    ax.set_xticks([-10, -5, 5, 10])
    ax.set_yticks([-1.0, -0.5, 0.5, 1.0])
    ax.tick_params(labelsize=18)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.tight_layout()
    plt.legend(loc=2, fontsize=18)
    plt.savefig("tanh.png")
    plt.show()


def plot_relu():
    x1 = np.linspace(-10.5, 0, 500)
    x2 = np.linspace(0, 10.5, 500)
    y1 = relu(x1)
    y2 = relu(x2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    # ax.spines['left'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.plot(x1, y1,  linestyle="-", color="darkviolet", linewidth=3)
    ax.plot(x2, y2, label='ReLU', linestyle="-", color="darkviolet")
    plt.xlim([-10.5, 10.5])
    plt.ylim([0, 10.02])
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.tick_params(labelsize=18)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    plt.tight_layout()
    plt.legend(loc=2, fontsize=18)
    plt.savefig("relu.png")
    plt.show()


def plot_prelu():
    x = np.linspace(-10.5, 10.5, 500)
    y = prelu(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    # ax.spines['left'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.plot(x, y, label='P-ReLU', linestyle="-", color="mediumvioletred")
    plt.xlim([-10.5, 10.5])
    plt.ylim([-1.2, 10.02])
    ax.set_xticks([-10, -5, 5, 10])
    ax.set_yticks([-1, 2, 4, 6, 8, 10])
    ax.tick_params(labelsize=18)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    plt.tight_layout()
    plt.legend(loc=2, fontsize=18)
    plt.savefig("prelu.png")
    plt.show()


def plot_elu():
    x = np.linspace(-10.5, 10.5, 500)
    y = elu(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    # ax.spines['left'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.plot(x, y, label='ELU', linestyle="-" )
    plt.xlim([-10.5, 10.5])
    plt.ylim([-1.2, 10.02])
    ax.set_xticks([-10, -5, 5, 10])
    ax.set_yticks([-1, 2, 4, 6, 8, 10])
    ax.tick_params(labelsize=18)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    plt.tight_layout()
    plt.legend(loc=2, fontsize=18)
    plt.savefig("elu.png")
    plt.show()


if __name__ == "__main__":
    # plot_sigmoid()
    # plot_tanh()
    # plot_relu()
    plot_prelu()
    plot_elu()

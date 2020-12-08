import math  #数学基本运算
import matplotlib.pyplot as plt #图形显示
import random  #随机数
import numpy as np #矩阵运算库
import pandas as pd #提供高性能易用数据类型和分析工具
import seaborn as sns #绘制数据分布，数据观察函数
from scipy.io import arff #方便导入arff文件数据
import sys #用于表示最大值和最小值


'''
该算法为BP神经网络算法
数据来源：iris.arff 鸢尾属植物的各项数据

数据内容包括：sepal length (花萼长度)、sepal width(花萼宽度)、
petal length(花瓣长度)、petal width(花瓣宽度)、class（种类）共三类

数据特征前四项为鸢尾属植物的特征为连续值，最后一项为植物标签，即所属类别为离散值
'''


def readingDatas():
    '''
    获取鸢尾属植物数据源
    :return:
    '''
    array = arff.loadarff("./Dataset/iris.arff")
    df = pd.DataFrame(array[0])
    df.insert(0, 'constant', 1)  # 添加constant列 值恒为1
    df = df.replace(b'Iris-setosa', 0)
    df = df.replace(b'Iris-versicolor', 1)
    df = df.replace(b'Iris-virginica', 2)
    data_array = np.array(df)
    dataSet = data_array.tolist()
    # print(dataSet)
    # print(len(dataSet))
    return dataSet

import math  #数学基本运算
import matplotlib.pyplot as plt #图形显示
import random  #随机数
import numpy as np #矩阵运算库
import pandas as pd #提供高性能易用数据类型和分析工具
import seaborn as sns #绘制数据分布，数据观察函数
from scipy.io import arff #方便导入arff文件数据
import sys #用于表示最大值和最小值
import datetime



'''
该算法为CART分类算法
数据来源：weather.nominal.arff



'''

def readingDatas():
    '''
    读入数据，并修改数据，添加一列数据且值恒为1，同时将最后一列枚举数据转换为0、1
    :return:
    '''
    array = arff.loadarff("./Dataset/weather.nominal.arff")
    df = pd.DataFrame(array[0])
    list = []
    list.append(df["outlook"].unique().tolist())
    list.append(df["temperature"].unique().tolist())
    list.append(df["humidity"].unique().tolist())
    list.append(df["windy"].unique().tolist())
    list.append(df["play"].unique().tolist())
    data_array = np.array(df)
    dataSet = data_array.tolist()
    return dataSet,list



def randomData(dataSet,rate):
    '''
    随机划分数据集为训练集和测试集
    :param dataSet:
    :param rate:
    :return:
    '''
    dataSetDemo = dataSet[:] #将数据存入另一个列表防止列表修改
    num = len(dataSetDemo)
    trainNum = int(rate*num)
    np.random.shuffle(dataSetDemo) #将列表随机乱序
    trainData = dataSetDemo[0:trainNum] #随机选取rate的数据成为分类数据
    testData = dataSetDemo[trainNum:num] #剩下的为测试数据
    return trainData,testData








def testDemo():
    print('test')
    dataSet,list = readingDatas()
    trainingData,testData = randomData(dataSet, 0.8)
    print(testData)
    print(list)




if __name__ == "__main__":
    testDemo()
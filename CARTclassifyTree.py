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
    classList = []
    classList.append(df["outlook"].unique().tolist())
    classList.append(df["temperature"].unique().tolist())
    classList.append(df["humidity"].unique().tolist())
    classList.append(df["windy"].unique().tolist())
    classList.append(df["play"].unique().tolist())
    data_array = np.array(df)
    dataSet = data_array.tolist()
    return dataSet,classList


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

def classifyGini(i,classList,trainingData):
    '''

    :param i: 计算第i个类别的gini
    :param classList: 输入类别列表
    :param trainingData: 输入数据
    :return:
    '''


    classListNum = []
    minGini = 1
    index = 0
    for ind,l in enumerate(classList[i]):
        # print(l)
        classify11 = 0
        classify12 = 0
        classify21 = 0
        classify22 = 0
        classNum = 0
        for data in trainingData:
            if data[i] == l:
                classNum += 1
                if data[-1] == b'yes':
                    classify11 += 1
                else:
                    classify12 += 1
            else:
                if data[-1] == b'yes':
                    classify21 += 1
                else:
                    classify22 += 1
        sumclass1 = classify11 + classify12
        sumclass2 = classify21 + classify22
        # print(classify2)
        gini1 = 1 - pow(classify11 / sumclass1, 2) - pow(classify12 /sumclass1, 2)
        gini2 = 1 - pow(classify21 / sumclass2, 2) - pow(classify22 / sumclass2, 2)
        gini = (classNum * gini1 + (len(trainingData)-classNum)*gini2)/len(trainingData)
        classListNum.append(gini)
        if gini < minGini:
            minGini = gini
            index = ind
    #     print(classNum)
    #     print('classify11:',classify11)
    #     print('classify12:', classify12)
    #     print('classify21:', classify21)
    #     print('classify22:', classify22)
    # print(classListNum)
    # print(minGini)
    # print(index)
    return minGini,index


def giniFunction(classList, trainingData):
    tempGini = []
    tempIndex = []
    for i,l in enumerate(classList):
        if i != 4:
            minGini,index =  classifyGini(i,classList,trainingData)
            tempGini.append(minGini)
            tempIndex.append(index)
    # a = np.array(tempGini)
    # b = np.array(tempIndex)
    # c = np.lexsort((b, a))
    # a.sort()
    # gini = [a,c]
    gini = [tempGini,tempIndex]
    return gini


def testFunction(gini,testData,classList):
    print(gini)
    # for data in testData:
    #     for index,cl in enumerate(classList):
    #         gini[1][index] = 0


def testDemo():
    print('test')
    dataSet,classList = readingDatas()
    trainingData,testData = randomData(dataSet, 0.8)
    print(testData)
    print(classList)
    gini = giniFunction(classList, trainingData)
    testFunction(gini, testData, classList)


if __name__ == "__main__":
    testDemo()
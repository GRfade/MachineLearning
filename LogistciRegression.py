import math  #数学基本运算
import matplotlib.pyplot as plt #图形显示
import random  #随机数
import numpy as np #矩阵运算库
import pandas as pd #提供高性能易用数据类型和分析工具
import seaborn as sns #绘制数据分布，数据观察函数
from scipy.io import arff #方便导入arff文件数据
import sys #用于表示最大值和最小值

'''
该算法为逻辑回归算法
数据来源：diabetes.arff 糖尿病人的各项数据
数据内容包括：preg（怀孕次数）、plas（葡萄糖浓度）、pres（血压）、skin（皮肤厚度）、insu（胰岛素）、mass（体重）、 pedi（谱系功能）、 age（年龄）、 class(是否为糖尿病人)
前8列为糖尿病人特征特征、最后一列为是否患有糖尿病
'''
def readingDatas():
    '''
    读入数据，并修改数据，添加一列数据且值恒为1，同时将最后一列枚举数据转换为0、1
    :return:
    '''
    array = arff.loadarff("./Dataset/diabetes.arff")
    df = pd.DataFrame(array[0])
    df.insert(0, 'constant', 1)  # 添加constant列 值恒为1
    df = df.replace(b'tested_negative', 0)
    df = df.replace(b'tested_positive', 1)
    data_array = np.array(df)
    dataSet = data_array.tolist()
    return dataSet


def normalization(dataSet):
    '''
    将数据进行归一化，提高准确率和效率
    将所有数据映射到[0,1]
    从第二列到第九列进行归一化 第一列值恒为1 不需要归一化，第十列为bool值不需要归一化
    :param dataSet:
    :return:
    '''
    maxList = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    minList = [float('inf'),float('inf'),float('inf'),float('inf'),
               float('inf'),float('inf'),float('inf'),float('inf')]
    for data in dataSet:
        for index, num in enumerate(data[1:-1]):
            if num > maxList[index]: maxList[index] = num
            elif num < minList[index]: minList[index] = num
    for data in dataSet:
        for index, num in enumerate(data[1:-1]):
            data[index+1] = (num - minList[index]) / (maxList[index]-minList[index])
    return dataSet


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
    trainData = dataSetDemo[0:trainNum] #随机选取80%的数据成为分类数据
    testData = dataSetDemo[trainNum:num] #剩下的为测试数据
    return trainData,testData


def decisionBoundary(data,coefficient):
    '''
    该函数为决策边界，即拟合的直线函数 z = θ_0x_0 + θ_1x_1 + ... + θ_nx_n
    :param data: 输入的数据
    :param coefficient: 线性回归系数
    :return: result 目标值
    '''
    data = data[:-1]
    vec1 = np.array(data)
    vec2 = np.array(coefficient)
    result = np.dot(vec1,vec2)
    return result


def hypothesisFunction(data,coefficient):
    '''
    假设函数,计算数据的分类
    :param data:
    :param coefficient:
    :return:
    '''
    boundary = decisionBoundary(data,coefficient) * (-1)
    result = 1 / (1 + math.exp( boundary ))
    return result

def  gradientDescent(trainData,rate,coefficient):
    '''
    梯度下降算法，本函数为一次迭代过程,有梯度下降就有学习率rate
    :param trainData:
    :param rate:
    :param coefficient:
    :return:
    '''
    coefficientDemo = coefficient[:]
    for index, num in enumerate(coefficient):  # 循环完毕就是一轮迭代
        sum = 0
        for data in trainData:
            sum += (hypothesisFunction(data, coefficient) - data[-1]) * data[index]
        sum = sum * rate / len(trainData)
        coefficientDemo[index] = num - sum
    coefficient = coefficientDemo[:]

    return coefficient


def costFunction(trainData,coefficient):
    '''
    计算代价函数,对数采用自然对数
    :param trainData:
    :param coefficient:
    :return:
    '''
    cost = 0
    for data in trainData:
        cost = data[-1] * math.log(hypothesisFunction(data,coefficient)) + (1-data[-1]) * math.log(1 - hypothesisFunction(data,coefficient))

    cost = (-1) * cost / len(trainData)
    return cost

def classify(trainData,rate):
    '''
    分类器，主要函数，通过分类器进行分类同时迭代参数不断优化
    :param trainData:
    :param rate:
    :return:
    '''
    coefficient = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    oldCost = 0.0
    i = 1
    newCost = costFunction(trainData, coefficient)
    while abs(oldCost - newCost) > 0.0000001 :
        coefficient = gradientDescent(trainData,rate,coefficient)

        oldCost = newCost
        newCost = costFunction(trainData, coefficient)
        print('第',i,'次迭代：')
        print('参数为：',coefficient)
        print('代价函数为：',newCost)
        i = i + 1
    print('训练集准确率为：',testLogistic(trainData,coefficient))
    return coefficient



def testLogistic(testData,coefficient):
    result = 0
    for data in testData:
        if hypothesisFunction(data,coefficient) >= 0.5 and data[-1] == 1: result = result +1
        elif hypothesisFunction(data,coefficient) < 0.5 and data[-1] == 0: result = result +1
    result = result / len(testData)
    return result


def testDemo():
    '''
    测试用例，测试逻辑回归算法
    :return:
    '''
    dataSet = readingDatas()
    normalization(dataSet)
    print(dataSet)
    trainData, testData = randomData(dataSet, 0.8)
    coefficient = classify(trainData, 0.8)
    print('测试集准确率为：',testLogistic(testData,coefficient))
    print('最终参数为：',coefficient)

if __name__ == "__main__":
    testDemo()

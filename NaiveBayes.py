import math  #数学基本运算
import matplotlib.pyplot as plt #图形显示
import random  #随机数
import numpy as np #矩阵运算库
import pandas as pd #提供高性能易用数据类型和分析工具
import seaborn as sns #绘制数据分布，数据观察函数
from scipy.io import arff #方便导入arff文件数据
import sys #用于表示最大值和最小值



'''
该算法为朴素贝叶斯算法 因为数据全为离散数据 故假设其符合多项式分布
数据来源：weather.nominal.arff 不同天气情况下是否可以出去玩的各项数据
数据内容包括：outlook temperature humidity windy play
所有数据全为离散值，前四项数据为天气情况，最后一项情况为是否可以出去玩
P(Y|X) = P(X|Y)P(Y) / P(X)
先验概率：P(Y) P(X)
条件概率P(X|Y)
后验概率：P(Y|X)
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
    trainData = dataSetDemo[0:trainNum] #随机选取80%的数据成为分类数据
    testData = dataSetDemo[trainNum:num] #剩下的为测试数据
    return trainData,testData



# def judge(property):
#     '''
#     将列名转换为下标index
#     :param property:
#     :return:
#     '''
#     index = -1
#     if property == 'outlook': index = 0
#     elif property == 'temperature': index = 1
#     elif property == 'humidity':index = 2
#     elif property == 'windy': index = 3
#     elif property == 'play':index = 4
#     return index


def priorProbability(data,list):
    '''
    先验概率计算,计算所有特征的先验概率 ,共有12个先验概率，3,3,2,2,2
    :param data:
    :param list:
    :return:
    '''
    probabilityList = [[],[],[],[],[]]
    for index,l in enumerate(list):
        for property in l:
            count = 0
            for d in data:
                if d[index] == property: count += 1
            probabilityList[index].append(count / len(data))
    return probabilityList








def conditionProbability(data,list):
    '''
    条件概率计算，计算所有特征的条件概率,2*（3+3+2+2）共有20个条件概率
    :param data:
    :param set:
    :return:
    '''
    conditionProbabilityList = [[[],[],[],[],[]],[[],[],[],[],[]]]
    demo = [b'yes',b'no'] #0是yes 1是no
    for num in range(2):
        for index, l in enumerate(list):
            for property in l:
                count = 0
                sum = 0
                for d in data:
                    if d[4] == demo[num]:
                        sum += 1
                        if d[index] == property: count += 1
                if sum != 0:
                    conditionProbabilityList[num][index].append(count / sum )
                else :
                    conditionProbabilityList[num][index].append(0)
    return conditionProbabilityList

def calculate(data,probabilityList,conditionProbabilityList,list,result):
    '''
    计算后验概率
    :return:
    '''

    rate = 1
    total = 1
    for i,property in enumerate(data):
        if property != b'yes' and property != b'no':
            rate = rate * conditionProbabilityList[result][i][list[i].index(property)]
            total = total * probabilityList[i][list[i].index(property)]
    rate = rate * probabilityList[4][list[i].index(property)]

    rate = (rate+1) / (total + 2) #引入拉普拉斯平滑
    return rate





def testNaiveBayes(testData,probabilityList,conditionProbabilityList,list):
    '''
    测试朴素贝叶斯算法,返回准确率
    :param testData:
    :param probabilityList:
    :param conditionProbabilityList:
    '''
    print('testNaiveBayes')
    rate = 0
    sum = len(testData)
    for data in testData:
        yesRate = calculate(data,probabilityList,conditionProbabilityList,list,0)
        if data[4] == b'yes' and yesRate >= 0.5: rate += 1
    rate = rate / sum
    print('准确率为：',rate)



def testDemo():
    # print("testDemo")
    dataSet,list =  readingDatas()
    trainData, testData =  randomData(dataSet,0.2)
    probabilityList = priorProbability(trainData,list)
    conditionProbabilityList = conditionProbability(trainData,list)
    testNaiveBayes(testData,probabilityList,conditionProbabilityList,list)


if __name__ == "__main__":
    testDemo()


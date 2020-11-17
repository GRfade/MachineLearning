import math  #数学基本运算
import matplotlib.pyplot as plt #图形显示
import random  #随机数
import numpy as np #矩阵运算库
import pandas as pd #提供高性能易用数据类型和分析工具
import seaborn as sns #绘制数据分布，数据观察函数

import sys #用于表示最大值和最小值

'''
该算法为线性回归算法
数据来源：abalone.data 鲍鱼的各项数据
数据内容包括：性别、长度、直径、高度、重量、肉重量、内脏重量、外壳重量、年龄
前8列为鲍鱼特征、最后一列为鲍鱼年龄（预测结果）
'''
def readingDatas():
    '''
    该函数利用pandas库导入data文件数据，并添加新的一列constant数据值为1，并将原第一列离散数据转换为数值
    转换方法为：对于同等地位的离散变量采取编码方式如'M', '1 0 0'，'I', '0 1 0'，'F', '0 0 1'
    这样就把一维id数据拓宽到三维从而解决了离散数据的问题
    最后将dataframe结构转换为list结构
    :return:dataSet list列表
    '''
    df = pd.read_csv("./Dataset/abalone.data",header=None)
    df.columns = ["Sex", "Length", "Diameter", "Height",
                  "Whole weight", "Shucked weight", "Viscera weight",
                  "Shell weight","Rings"]

    df.insert(0, 'constant', 1)  #添加constant列 值恒为1
    df = df.replace('M', '1 0 0')  #对sex列的数据进行连续化处理
    df = df.replace('I', '0 1 0')
    df = df.replace('F', '0 0 1')
    df['Sex1'] = df['Sex'].map(lambda x:x.split(' ')[0])
    df['Sex2'] = df['Sex'].map(lambda x:x.split(' ')[1])
    df['Sex3'] = df['Sex'].map(lambda x:x.split(' ')[2])
    df = df.drop(['Sex'], axis=1)   # 删除Sex列
    # print(df)
    columns = list(df)
    # print(columns)
    columns.insert(1,columns.pop(columns.index('Sex1')))
    columns.insert(2, columns.pop(columns.index('Sex2')))
    columns.insert(3, columns.pop(columns.index('Sex3')))
    # print(columns)
    df = df.loc[:, columns]
    df = df.replace('1', 1)
    df = df.replace('0', 0)

    dataSet = np.array(df).tolist()
    for data in dataSet:
        data[-1] = data[-1] + 1.5
    # print(dataSet)
    return dataSet


def randomData(dataSet,rate):
    '''
    将数据随机分为训练数据和测试数据
    :param dataSet:数据集
    :param rate:比率
    :return: trainingSet,testSet
    '''
    datasetDemo = dataSet[:] #将数据存入另一个列表防止列表修改
    num = len(datasetDemo)
    trainNum = int(rate * num)
    np.random.shuffle(datasetDemo) #将列表随机乱序
    trainData = datasetDemo[0:trainNum] #随机选取80%的数据成为分类数据
    testData = datasetDemo[trainNum:num] #剩下的为测试数据
    return trainData,testData


def  gradientDescent(trainData,rate,coefficient):
    '''
    梯度下降优化算法，该函数为一轮迭代过程
    :param trainDataNoLabel:
    :param rate:
    :return:
    '''
    coefficientDemo = coefficient[:]
    for index,num in enumerate(coefficient):  # 循环完毕就是一轮迭代
        sum = 0
        for data in trainData:
            sum += ( hypothesisFunction(data,coefficient) - data[-1] ) * data[index]
        sum = sum * rate / len(trainData)
        coefficientDemo[index] = num - sum
    coefficient = coefficientDemo[:]
    return coefficient


def hypothesisFunction(data,coefficient):
    '''
    该函数为假设函数，即拟合的直线函数 h(x) = θ_0x_0 + θ_1x_1 + ... + θ_nx_n
    :param data: 输入的数据
    :param coefficient: 线性回归系数
    :return: result 目标值
    '''
    data = data[:-1]
    vec1 = np.array(data)
    vec2 = np.array(coefficient)
    result = np.dot(vec1,vec2)

    return result


def trainLinear(trainData,rate):
    '''
    训练线性回归算法，获得最小代价函数，利用梯度下降算法,进行n轮迭代
    :param trainDate:测试数据
    :param rate:学习率(0~1)
    :return:coefficient 回归系数
    '''
    print("linearRegression")
    coefficient = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]  # 系数列表初始化
    oldCost = 0.0
    newCost = costFunction(trainData, coefficient)
    while abs(oldCost - newCost) > 0.01 : #总迭代次数
        coefficient = gradientDescent(trainData,rate,coefficient)
        oldCost = newCost
        newCost = costFunction(trainData, coefficient)
        print('代价函数值为：',newCost)

    return coefficient


def costFunction(trainData,coefficient):
    cost = 0
    for data in trainData:
        cost += np.square(hypothesisFunction(data, coefficient) - data[-1])
    cost = cost / (len(trainData)*2)
    return cost

def Testlinea(testData,coefficient):
    '''
    测试算法，利用迭代得到的回归系数，进行测试，并绘制图像
    :param testData:
    :param coefficient:
    :return:
    '''
    print(Testlinea)


def testLinear():
    '''
    测试线性回归算法
    '''
    dataSet = readingDatas()
    trainData,testData = randomData(dataSet,0.8) #获取训练数据和测试数据
    coefficient = trainLinear(trainData,0.5)
    print(coefficient)



if __name__ == "__main__":
    testLinear()


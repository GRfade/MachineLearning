import math  #数学基本运算
import matplotlib.pyplot as plt #图形显示
import random  #随机数
import numpy as np #矩阵运算库
import pandas as pd #提供高性能易用数据类型和分析工具
import seaborn as sns #绘制数据分布，数据观察函数
from scipy.io import arff #方便导入arff文件数据
import sys #用于表示最大值和最小值



'''
该算法为softmaxRegression算法（多项逻辑回归算法） 
数据来源：iris.arff 鸢尾属植物的各项数据

数据内容包括：sepal length (花萼长度)、sepal width(花萼宽度)、
petal length(花瓣长度)、petal width(花瓣宽度)、class（种类）共三类

数据特征前四项为鸢尾属植物的特征，最后一项为植物标签，即所属类别
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

def picture(dataSet):
    '''
    绘制数据集中特征与标签之间的关系
    :param dataSet:
    :return:
    '''
    array = np.array(dataSet)
    plt.figure(1)
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    plt.sca(ax1)
    plt.xlabel('sepal length', fontsize=15, color='r')
    plt.scatter(array[:,0], array[:,-1],alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    plt.sca(ax2)
    plt.xlabel('sepal width', fontsize=15, color='r')
    plt.scatter(array[:,1], array[:,-1],alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    plt.sca(ax3)
    plt.xlabel('petal length', fontsize=15, color='r')
    plt.scatter(array[:,2], array[:, -1], alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    plt.sca(ax4)
    plt.xlabel('petal width', fontsize=15, color='r')
    plt.scatter(array[:,3], array[:, -1], alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    plt.show()

def decisionBoundary(data,coefficient):
    '''
    决策边界即拟合的直线函数 z = θ_0x_0 + θ_1x_1 + ... + θ_nx_n 所对应的向量(三维列表)
    :param data: 输入的数据
    :param coefficient: 线性回归系数
    :return: result 目标值
    '''
    result = []
    data = data[:-1]
    vec1 = np.array(data)
    for coe in coefficient:
        vec2 = np.array(coe)
        result.append(np.dot(vec1, vec2))
    return result

def hypothesisFunction(data,coefficient):
    '''
    假设函数,计算数据的分类,j假设函数结果为一个分类向量，其中数值最大的为训练结果分类
    :param data:
    :param coefficient:
    :return:
    '''
    result = []
    boundary = decisionBoundary(data, coefficient)
    sum = 0
    for index in range(len(coefficient)):
        demo = math.exp(boundary[index])
        result.append(demo)
        sum += demo
    for index in range(len(result)):
        result[index] = result[index] / sum
    return result


def costFunction(trainData,coefficient):
    cost = 0
    for data in trainData:
        if data[-1] == 0:
            cost += math.log(hypothesisFunction(data,coefficient)[0])
        elif data[-1] == 1:
            cost += math.log(hypothesisFunction(data,coefficient)[1])
        elif data[-1] == 2:
            cost += math.log(hypothesisFunction(data,coefficient)[2])
    cost = (-1) * cost / len(trainData)

    return cost


def  gradientDescent(trainData,rate,coefficient):
    '''
    梯度下降算法
    :param trainData:
    :param rate:
    :param coefficient:
    :return:
    '''
    print('gradientDescent')
    coefficientDemo = coefficient[:]
    for index1,coe in enumerate(coefficient):#迭代三个分类对应的参数向量
        for index2,num in enumerate(coe):
            sum = 0
            for data in trainData:
                result = hypothesisFunction(data, coefficient)
                if data[-1] == index1:
                    sum += (1 - result[index1]) * data[index2]
                else:
                    sum += (0 - result[index1]) * data[index2]

            sum = sum * rate / len(trainData)
            coefficientDemo[index1][index2] = num + sum
        coefficient = coefficientDemo[:]

    return coefficient


def classify(trainData,rate):

    coefficient = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]
    oldCost = 0.0
    i = 1
    newCost = costFunction(trainData, coefficient)
    while abs(oldCost - newCost) > 0.00001:
        coefficient = gradientDescent(trainData, rate, coefficient)

        oldCost = newCost
        newCost = costFunction(trainData, coefficient)
        print('第', i, '次迭代：')
        print('参数为：', coefficient)
        print('代价函数为：', newCost)
        print('训练集准确率为：', testSoftmaxRegression(trainData, coefficient))
        i = i + 1

    return coefficient


def testSoftmaxRegression(testData,coefficient):

    rata = 0
    for data in testData:
        result = hypothesisFunction(data, coefficient)
        if data[-1] == result.index(max(result)): rata += 1

    rata = rata / len(testData)
    return rata




def testDemo():
    dataSet = readingDatas()
    trainData,testData = randomData(dataSet, 0.8)
    # picture(dataSet)
    coefficient = classify(trainData, 0.3)
    rata = testSoftmaxRegression(testData,coefficient)
    print('Softmax算法准确率为：',rata)

if __name__ == "__main__":
    testDemo()

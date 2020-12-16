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
    df.columns = ["sepallength","sepalwidth","petallength","petalwidth","class"]
    df = df.replace(b'Iris-setosa', '1 0 0')
    df = df.replace(b'Iris-versicolor', '0 1 0')
    df = df.replace(b'Iris-virginica', '0 0 1')
    df['class1'] = df['class'].map(lambda x:x.split(' ')[0])
    df['class2'] = df['class'].map(lambda x:x.split(' ')[1])
    df['class3'] = df['class'].map(lambda x:x.split(' ')[2])
    df = df.drop(['class'], axis=1)  # 删除Sex列
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
    trainData = dataSetDemo[0:trainNum] #随机选取rate的数据成为分类数据
    testData = dataSetDemo[trainNum:num] #剩下的为测试数据
    return trainData,testData


def testDemo():
    print('test')
    dataSet = readingDatas()
    trainingData,testData = randomData(dataSet, 0.8)
    print(testData)



if __name__ == "__main__":
    testDemo()
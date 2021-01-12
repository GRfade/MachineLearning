# import math  #数学基本运算
import matplotlib.pyplot as plt #图形显示
# import random  #随机数
import numpy as np #矩阵运算库
import pandas as pd #提供高性能易用数据类型和分析工具
import seaborn as sns #绘制数据分布，数据观察函数
from scipy.io import arff #方便导入arff文件数据
# import sys #用于表示最大值和最小值
# import datetime





'''
该算法为PCA算法，主程序分析算法
本次算法调用数据：diabetes.arff 糖尿病人的各项数据
前八列为数据，最后一列为标签

'''

def readingDatas():
    '''
    读入数据
    :return:data_array 类别是 numpy.ndarray
    '''
    array = arff.loadarff("./Dataset/diabetes.arff")
    df = pd.DataFrame(array[0])
    data_array = np.array(df)
    lableList = data_array.T[-1]
    data_array = np.delete(data_array, -1, axis=1) #s删除最后一列
    return data_array,lableList



def zeroMean(dataSet):
    '''
    将数据矩阵减去每一个特征的均值
    :param dataSet:类别是 numpy.ndarray，是数据集
    :return: newData类别是 numpy.ndarray
    '''
    meanVal = np.mean(dataSet, axis=0)  # 按列求均值，即求各个特征的均值
    newData = dataSet - meanVal #二维矩阵减去一维矩阵
    return newData


def covarianceMatrix(dataSet):
    '''
    求出协方差矩阵，为n*n的矩阵，n为特征维度
    :param dataSet:输入为数据矩阵
    :return:covariance:返回协方差为矩阵
    '''
    # covariance = np.dot(dataSet.T , dataSet) #没有除以m
    covariance = np.cov(dataSet.T.astype(float))

    return covariance

def eigenValuesVectors(covariance,k):
    '''
        计算对应的特征值和特征向量,并将特征值按照从大到小排序
    :param covariance: 协方差矩阵
    :param k: 取前k行
    :return:
    '''
    vals,vecs = np.linalg.eigh(covariance)
    eigen = zip(vals,vecs)
    eigen = sorted(eigen,reverse=True)
    # for i in eigen:
    #     print(i)
    eigen = list(zip(*eigen))
    # print(eigen)
    vals = eigen[0]
    vecs = eigen[1]
    basisVector =vecs[:k]
    basisVector = np.array(basisVector, dtype=float)
    return basisVector

def pca(dataSet,k):
    '''
    PCA算法，
    :param dataSet:
    :param k:
    :return:
    '''

    covariance = covarianceMatrix(zeroMean(dataSet))
    basisVector = eigenValuesVectors(covariance, k)
    data = np.dot(basisVector,dataSet.T).T
    return data

def pltshow1(data,labelList):
    '''
    降维后数据展示，展示降维到二维时的状态
    :return:
    '''

    # dataList = list(zip(data, lableList))

    colors = labelList
    colors[colors == b'tested_positive'] = 'r'
    colors[colors == b'tested_negative'] = 'b'
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.set_title('Scatter Plot')
    # ax1.scatter(data.T[0] c=colors, marker='o')

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Scatter Plot')
    print(data.T)
    temp =  a=[1]*len(data)

    print(data.T)
    ax1.scatter(data.T[0], temp, c=colors, marker='o')

    plt.show()


def pltshow2(data,labelList):
    '''
    降维后数据展示，展示降维到二维时的状态
    :return:
    '''

    # dataList = list(zip(data, lableList))

    colors = labelList
    colors[colors == b'tested_positive'] = 'r'
    colors[colors == b'tested_negative'] = 'b'
    # print(colors)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Scatter Plot')
    ax1.scatter(data.T[0],data.T[1], c=colors, marker='o')
    plt.show()



def main():
    '''
    测试PCA算法，
    '''
    dataSet, labelList = readingDatas()

    data1 = pca(dataSet, 1)
    print('PCA算法降维到一维：\n', data1)
    pltshow1(data1, labelList)

    data2 = pca(dataSet,2)
    print('PCA算法降维到二维：\n', data2)
    pltshow2(data2,labelList)





if __name__ == "__main__":
    print("PCA算法：")
    main()



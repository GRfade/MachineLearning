# import math  #数学基本运算
# import matplotlib.pyplot as plt #图形显示
# import random  #随机数
import numpy as np #矩阵运算库
import pandas as pd #提供高性能易用数据类型和分析工具
# import seaborn as sns #绘制数据分布，数据观察函数
from scipy.io import arff #方便导入arff文件数据
# import sys #用于表示最大值和最小值
# import datetime





'''
该算法为PCA算法，主程序分析算法

'''

def readingDatas():
    '''
    读入数据，并修改数据，添加一列数据且值恒为1，同时将最后一列枚举数据转换为0、1
    :return:data_array 类别是 numpy.ndarray
    '''
    array = arff.loadarff("./Dataset/diabetes.arff")
    df = pd.DataFrame(array[0])
    data_array = np.array(df)

    data_array = np.delete(data_array, -1, axis=1) #s删除最后一列
    return data_array



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
    covariance = np.dot(dataSet.T , dataSet)
    print(covariance)
    return covariance

def main():
    dataSet = readingDatas()
    dataSet = zeroMean(dataSet)
    covariance = covarianceMatrix(dataSet)
    print(covariance)





if __name__ == "__main__":
    print("PCA算法：")
    main()


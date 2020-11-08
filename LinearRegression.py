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
    该函数利用pandas库导入data文件数据，转换为DataFrame数据结构
    :return:返回data列表  pandas.core.frame.DataFrame
    '''
    dataSet = pd.read_csv("./Dataset/abalone.data",header=None)
    dataSet.columns = ["Sex", "Length", "Diameter", "Height",
                  "Whole weight", "Shucked weight", "Viscera weight",
                  "Shell weight","Rings"]
    dataSet.insert(0, 'constant', 1)
    return dataSet


def randomData(dataSet,rate):
    '''
    将数据随机分为训练数据和测试数据
    :param dataSet:
    :param rate:
    :return:
    '''




def testLinear():
    '''
    测试线性回归算法
    '''
    dataSet = readingDatas()
    print(dataSet)



if __name__ == "__main__":
    testLinear()
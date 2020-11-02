import math  #数学基本运算
import matplotlib.pyplot as plt #图形显示
import random  #随机数
import numpy as np #矩阵运算库
import pandas as pd #提供高性能易用数据类型和分析工具
import seaborn as sns #绘制数据分布，数据观察函数
from scipy.io import arff #方便导入arff文件数据



'''
本算法为KNN算法。KNearestNeighbor
可采用归一化数据方式，将所有数据范围压缩到0~1
本次算法调用数据：diabetes.arff 糖尿病人的各项数据
数据内容包括：preg（怀孕次数）、plas（葡萄糖浓度）、pres（血压）、skin（皮肤厚度）、insu（胰岛素）、mass（体重）、 pedi（谱系功能）、 age（年龄）、 class(是否为糖尿病人)
前8列为特征、最后一列为
'''

#读入数据
def readingDatas():
    data = arff.loadarff("./Dataset/diabetes.arff")
    df = pd.DataFrame(data[0])
    # print(data)
    print(df)
    print(type(df))



##不需要归一化
def transfun(data):
    '''将data归一化，'''
    sum = np.sum(np.std(data, axis=0))
    mean = data.mean(axis=0)
    data = (data -mean) / sum
    return data
#定义KNN分类器函数
#函数参数包括：（测试数据，训练数据，分类，k值）
def KNNclassify(inX,dataSet, labels, k):
    print("classify")


def testKNN():
    print("test")

if __name__ == "__main__":
    testKNN()
    readingDatas()


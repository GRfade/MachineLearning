import math  #数学基本运算
import matplotlib.pyplot as plt #图形显示
import random  #随机数
import numpy as np #矩阵运算库
import pandas as pd #提供高性能易用数据类型和分析工具
import seaborn as sns #绘制数据分布，数据观察函数
from scipy.io import arff #方便导入arff文件数据



'''
本算法为KNN算法。KNearestNeighbor 监督学习当中的分类算法
可采用归一化数据方式，将所有数据范围压缩到0~1
采用欧式距离计算
本次算法调用数据：diabetes.arff 糖尿病人的各项数据
数据内容包括：preg（怀孕次数）、plas（葡萄糖浓度）、pres（血压）、skin（皮肤厚度）、insu（胰岛素）、mass（体重）、 pedi（谱系功能）、 age（年龄）、 class(是否为糖尿病人)
前8列为特征、最后一列为是否患有糖尿病
'''

#读入数据
def readingDatas():
    #从diabetes.arff文件中导入数据，最终转化为两个，一个列表为特征值，一个列表为分类结果
    array = arff.loadarff("./Dataset/diabetes.arff")
    df = pd.DataFrame(array[0])

    data_array = np.array(df)
    data_list = data_array.tolist()
    data = []
    results = []
    for list in data_list:
        data.append(list[:-1])
        results.append(list[-1])
    return data,results

#计算两个数据之间的欧式距离
def distance(datalist,target):
    dist = []
    vec1 = np.array(target)
    for data in range(0,len((datalist))):
        vec2 = np.array(data)
        dist.append(np.linalg.norm(vec1-vec2))
    return dist


def transfun(data):
    '''将data归一化，'''
    sum = np.sum(np.std(data, axis=0))
    mean = data.mean(axis=0)
    data = (data -mean) / sum
    return data


#定义KNN分类器函数
#函数参数包括：（测试数据，训练数据，分类，k值）
def KNNclassify(dataset, labels, k):
    print("classify")



def testKNN():
    print("testKNN")


if __name__ == "__main__":
    testKNN()
    data,results = readingDatas()
    # dist = distance(data,[0,1,2,3,4,5,6,7])
    # print(len(dist))


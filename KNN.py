import math  #数学基本运算
import matplotlib.pyplot as plt #图形显示
import random  #随机数
import numpy as np #矩阵运算库
import pandas as pd #提供高性能易用数据类型和分析工具
import seaborn as sns #绘制数据分布，数据观察函数

'''
本算法为KNN算法。
采用归一化数据方式，将所有数据范围压缩到0~1
'''

#读入数据
def readingDatas():
    #传入数据：无
    # 传出数据：列表 [['str','str'......],['str','str']......]
    with open("./Dataset/abalone.data") as file:
        contents = file.read()        #print(type(contents)) 数据格式为str
    contents = contents.split('\n')    # print(type(content)) #输出内容为list，list内部为str
    datas = []
    results = []
    #将数据分为datas第一列离散数据，第二列至第八列float数据
    # results：最后一列分类结果数据
    for content in contents:
        data = content.split(',')
        results.append(int(data[-1]))
        del data[-1]
        for i in range(1,len(data)):
            data[i] = float(data[i])
        datas.append(data)
    return datas,results


#定义KNN分类器函数
#函数参数包括：（测试数据，训练数据，分类，k值）
def KNNclassify(inX,dataSet, labels, k):
    print("classify")


def testKNN():
    print("test")

if __name__ == "__main__":
    testKNN()
    datas,results = readingDatas()
    print(datas)
    # print(results)

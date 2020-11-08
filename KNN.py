# import math  #数学基本运算
# import matplotlib.pyplot as plt #图形显示
# import random  #随机数
import numpy as np #矩阵运算库
import pandas as pd #提供高性能易用数据类型和分析工具
# import seaborn as sns #绘制数据分布，数据观察函数
from scipy.io import arff #方便导入arff文件数据
import sys #用于表示最大值和最小值





'''
本算法为KNN算法。KNearestNeighbor 监督学习当中的分类算法
KNN可用于分类算法，也可用于回归算法
KNN算法是一种消极算法（没有训练的过程，到了要决策的时候才会利用已有的数据进行决策）
可采用归一化数据方式，将所有数据范围压缩到0~1
采用欧式距离计算
本次算法调用数据：diabetes.arff 糖尿病人的各项数据
数据内容包括：preg（怀孕次数）、plas（葡萄糖浓度）、pres（血压）、skin（皮肤厚度）、insu（胰岛素）、mass（体重）、 pedi（谱系功能）、 age（年龄）、 class(是否为糖尿病人)
前8列为特征、最后一列为是否患有糖尿病
'''

#读入数据
def readingDatas():
    #从diabetes.arff文件中导入数据，最终转化为前8列为特征、最后一列为分类结果
    array = arff.loadarff("./Dataset/diabetes.arff")
    df = pd.DataFrame(array[0])

    data_array = np.array(df)
    data_list = data_array.tolist()
    # print(type(data_list))
    # data = []
    # results = []
    # for list in data_list:
    #     data.append(list[:-1])
    #     results.append(list[-1])

    return data_list

#将数据随机分为训练数据和测试数据
def randomData(dataset,rate):
    datasetDemo = dataset[:] #将数据存入另一个列表防止列表修改
    num = len(datasetDemo)
    trainNum = int(rate*num)
    np.random.shuffle(datasetDemo) #将列表随机乱序
    trainData = datasetDemo[0:trainNum] #随机选取80%的数据成为分类数据
    testData = datasetDemo[trainNum:num] #剩下的为测试数据
    return trainData,testData



#计算两个数据之间的欧式距离
def distance(datalist,target):
    #datalist为列表中的列表
    target = target[:-1]
    dist = []
    vec1 = np.array(target)
    for data in datalist:
        vec2 = np.array(data[:-1])
        dist.append(np.linalg.norm(vec1-vec2))
    return dist


#归一化数据，将数据范围调控到0~1之间
# def transfun(data):
#     '''将data归一化，'''
#     sum = np.sum(np.std(data, axis=0))
#     mean = data.mean(axis=0)
#     data = (data -mean) / sum
#     return data

#选取前K个distance最小的值，故采取对整个列表排序
#采取选择排序的思想遍历选取最小的前k个值
#返回的mins 为最小值对应的下标列表
def select(distance,k):
    distance = distance[:] #创建一个副本
    n = len(distance)
    mins = []
    for i in range(0,k):
        min = float('inf')  # 表示最大浮点数
        index = 0
        for j in range(0,n):
            if distance[j] < min :
                min = distance[j]
                index = j
        distance[index] = sys.maxsize #表示最大整数
        mins.append(index)
        # print(distance)
    return mins



#定义KNN分类器函数
#函数参数包括：（总体数据，分类，k值）
def classifyKNN(trainData, testData, k):
    right = 0
    wrong = 0
    dist = []# 创建欧式距离集合
    for test in testData:
        dist = distance(trainData,test) #获取每一个测试数据的欧氏距离列表
        min = select(dist,k)  #选出最小的欧氏距离对应的下标

        #得出测试数据的结果
        dict = {'tested_positive':0,'tested_negative':0}
        for i in min:
            if trainData[i][-1] == b'tested_positive' :
                dict['tested_positive'] = dict['tested_positive'] + 1
            else :
                dict['tested_negative'] = dict['tested_negative'] + 1
            # print(dict)

        if dict['tested_positive'] > dict['tested_negative'] : #判断患有糖尿病
            if test[-1] == b'tested_positive' :#测试准确
                right += 1
        elif dict['tested_positive'] <= dict['tested_negative'] : #判断不患有糖尿病
            if test[-1] == b'tested_negative' :#测试准确
                right += 1
    accuracy = right/len(testData)

    return accuracy








    # random.seed(len(dataset)) #设置种子
    # classifyData = random.sample(dataset, classifyNum)




def testKNN():
    print("testKNN")
    dataset = readingDatas()
    # print(len(dist))
    trainData,testData = randomData(dataset,0.8)
    accuracy = classifyKNN(trainData, testData, 10)
    print("准确率为：",accuracy)




if __name__ == "__main__":
    testKNN()



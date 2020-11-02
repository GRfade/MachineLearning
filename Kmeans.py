import math  #数学基本运算
import matplotlib.pyplot as plt #图形显示
import random  #随机数
import numpy as np #矩阵运算库
import pandas as pd #提供高性能易用数据类型和分析工具

"""
本算法调用数据为 abalone.data文件
1.先将文件数据读取，转换为列表，列表内每一项为列表，包括
Sex Length	Diam	Height	Whole	Shucked	Viscera	Shell	Rings 这些数据

2.再对这些数据进行处理

3.输出图形化数据
"""


def readingDatas():
    # 打开文件，读取数据，改变格式为列表,去除第一列数据，该列数据不适合做Kmeans算法
    #传入数据：无
    # 传出数据：列表 [['str','str'......],['str','str']......]
    with open("./Dataset/abalone.data") as file:
        contents = file.read()        #print(type(contents)) 数据格式为str

    contents = contents.split('\n')    # print(type(content)) #输出内容为list，list内部为str
    datas = []
    for content in contents:
        data = content.split(',')
        del data[0]
        datas.append(data)
    return datas

def initCentroids(dataSet, k):
    #在数据集中随机选取K个数据
    #传入数据：[['str','str'......],['str','str']......] int
    #传出数据为：[['str','str'......],['str','str']......]
    return random.sample(dataSet, k)

def minDistance(dataSet, centroidList):
    #计算最小数据集，分别属于哪些数据集，输入为总数据集，和K个均值，输出为字典，10个簇
    #传入数据：[['str','str'......],['str','str']......]，[['str','str'......],['str','str']......]
    #传出数据：[['str','str'......],['str','str']......]
    # clusterDict = {0:None,1:None,2:None,3:None,4:None,5:None,6:None,7:None,8:None,9:None}  # dict保存簇类结果
    clusterDict = {}
    k = len(centroidList)
    for item in dataSet:
        vec1 = np.array(item, dtype = float)#转变为numpy数组
        flag = -1
        minDis = float("inf")  # 初始化为最大值
        for i in range(k):
            vec2 = np.array(centroidList[i] , dtype = float) #转变为numpy数组,格式为float
            distance = np.linalg.norm(vec1 - vec2) #计算数据之间欧式距离借助numpy库
            if distance < minDis:
                minDis = distance
                flag = i  # 循环结束时， flag保存与当前item最近的蔟标记
        if flag not in clusterDict.keys():
            clusterDict.setdefault(flag, [])
        clusterDict[flag].append(item)  # 加入相应的类别中
    return clusterDict  # 不同的类别





def getCentroids(clusterDict):
    #重新计算k个质心
    # 传入数据:字典
    # 传出数据为列表
    centroidList = []
    for key in clusterDict.keys():
        # print(key)
        clusterDict[key] = np.array(clusterDict[key],dtype = float)
        centroid = np.mean(clusterDict[key], axis=0)
        centroid = centroid .tolist()
        # print(type(centroid))
        centroidList.append(centroid)
    return centroidList  #得到新的质心


def getVar(centroidList, clusterDict):
    # 计算各蔟集合间的均方误差
    # 将蔟类中各个向量与质心的距离累加求和
    sum = 0.0
    for key in clusterDict.keys():
        vec1 = np.array(centroidList[key],dtype = float)
        distance = 0.0
        for item in clusterDict[key]:
            vec2 = np.array(item,dtype = float)
            distance += np.linalg.norm(vec1 - vec2)
        sum += distance
    return sum


def showCluster(centroidList, clusterDict):
    # 展示聚类结果
    #调用matplotlib.pyplot库
    colorMark = ['or', 'ob', 'og', 'ok', 'oy', 'ow'] #不同簇类标记，o表示圆形，另一个表示颜色
    centroidMark = ['dr', 'db', 'dg', 'dk', 'dy', 'dw']

    for key in clusterDict.keys():
        plt.plot(centroidList[key][0], centroidList[key][1], centroidMark[key], markersize=12) #质心点
        for item in clusterDict[key]:
            plt.plot(item[0], item[1], colorMark[key])
    plt.show()


def testkmeans():
    plt.cla()
    dataSet = readingDatas() #读入数据
    centroidList = initCentroids(dataSet,6) #初始化K个类
    clusterDict = minDistance(dataSet, centroidList) #进行第一轮分配
    newVar = getVar(centroidList, clusterDict) #计算当前均方误差
    oldVar = 1 #目标均方误差
    times = 2
    while abs(newVar - oldVar) >= 0.00001: #当误差小于0.00001结束
        centroidList = getCentroids(clusterDict) #重新计算质点
        clusterDict = minDistance(dataSet, centroidList) #重新分配
        oldVar = newVar
        newVar = getVar(centroidList, clusterDict)
        times += 1
        showCluster(centroidList, clusterDict)

    for key in clusterDict.keys():
        print(len(clusterDict[key]))





if __name__ == "__main__":
    testkmeans()
    # dataSet = readingDatas()
    # centroidList = initCentroids(dataSet, 10)
    # clusterDict = minDistance(dataSet,centroidList)
    # a = getCentroids(clusterDict)

    # centroidList = getCentroids(clusterDict)






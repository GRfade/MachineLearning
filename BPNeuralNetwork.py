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
该算法为BP神经网络算法
数据来源：iris.arff 鸢尾属植物的各项数据

数据内容包括：sepal length (花萼长度)、sepal width(花萼宽度)、
petal length(花瓣长度)、petal width(花瓣宽度)、class（种类）共三类

数据特征前四项为鸢尾属植物的特征为连续值，最后一项为植物标签，即所属类别为离散值

构建一个具有1个隐藏层的神经网络，隐层的大小为10
输入层为4个特征，输出层为3个分类
(1,0,0)为第一类，(0,1,0)为第二类，(0,0,1)为第三类
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


# def layerOutput(parameter,input):
#     '''
#     每一层的输出，结果为向量
#     :param parameter:
#     :param input:
#     :return:
#     '''
#
#     return 0

# def hypothesisFunction(data,parameter):
#     '''
#     假设函数的输出，结果为K维向量
#     :param data:
#     :param parameter:
#     :return:
#     '''
#     result = 0
#     return result




# 1.初始化参数
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)

    # 权重和偏置矩阵
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    # 通过字典存储参数
    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return parameters


# 2.前向传播
def forward_propagation(X, parameters):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    # 通过前向传播来计算a2
    z1 = np.dot(w1, X) + b1     # 这个地方需注意矩阵加法：虽然(w1*X)和b1的维度不同，但可以相加
    a1 = np.tanh(z1)            # 使用tanh作为第一层的激活函数
    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))  # 使用sigmoid作为第二层的激活函数

    # 通过字典存储参数
    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}

    return a2, cache


# 3.计算代价函数
def compute_cost(a2, Y):
    m = Y.shape[1]      # Y的列数即为总的样本数

    # 采用交叉熵（cross-entropy）作为代价函数
    logprobs = np.multiply(np.log(a2), Y) + np.multiply((1 - Y), np.log(1 - a2))
    cost = - np.sum(logprobs) / m

    return cost


# 4.反向传播（计算代价函数的导数）
def backward_propagation(parameters, cache, X, Y):
    m = Y.shape[1]

    w2 = parameters['w2']

    a1 = cache['a1']
    a2 = cache['a2']

    # 反向传播，计算dw1、db1、dw2、db2
    dz2 = a2 - Y
    dw2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.multiply(np.dot(w2.T, dz2), 1 - np.power(a1, 2))
    dw1 = (1 / m) * np.dot(dz1, X.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    grads = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}

    return grads


# 5.更新参数
def update_parameters(parameters, grads, learning_rate=0.4):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    dw1 = grads['dw1']
    db1 = grads['db1']
    dw2 = grads['dw2']
    db2 = grads['db2']

    # 更新参数
    w1 = w1 - dw1 * learning_rate
    b1 = b1 - db1 * learning_rate
    w2 = w2 - dw2 * learning_rate
    b2 = b2 - db2 * learning_rate

    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return parameters


# 建立神经网络
def nn_model(X, Y, n_h, n_input, n_output, num_iterations=10000, print_cost=False):
    np.random.seed(3)

    n_x = n_input           # 输入层节点数
    n_y = n_output          # 输出层节点数

    # 1.初始化参数
    parameters = initialize_parameters(n_x, n_h, n_y)

    # 梯度下降循环
    for i in range(0, num_iterations):
        # 2.前向传播
        a2, cache = forward_propagation(X, parameters)
        # 3.计算代价函数
        cost = compute_cost(a2, Y)
        # 4.反向传播
        grads = backward_propagation(parameters, cache, X, Y)
        # 5.更新参数
        parameters = update_parameters(parameters, grads)

        # 每1000次迭代，输出一次代价函数
        if print_cost and i % 1000 == 0:
            print('迭代第%i次，代价函数为：%f' % (i, cost))

    return parameters


# 6.模型评估
def predict(parameters, x_test, y_test):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    z1 = np.dot(w1, x_test) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))

    # 结果的维度
    n_rows = y_test.shape[0]
    n_cols = y_test.shape[1]

    # 预测值结果存储
    output = np.empty(shape=(n_rows, n_cols), dtype=int)

    for i in range(n_rows):
        for j in range(n_cols):
            if a2[i][j] > 0.5:
                output[i][j] = 1
            else:
                output[i][j] = 0

    print('预测结果：', output)
    print('真实结果：', y_test)

    count = 0
    for k in range(0, n_cols):
        if output[0][k] == y_test[0][k] and output[1][k] == y_test[1][k] and output[2][k] == y_test[2][k]:
            count = count + 1
        else:
            print('错误分类样本的序号：', k + 1)

    acc = count / int(y_test.shape[1]) * 100
    print('准确率：%.2f%%' % acc)

    return output


# 7.结果可视化
# 特征有4个维度，类别有1个维度，一共5个维度，故采用了RadViz图
def result_visualization(x_test, y_test, result):
    cols = y_test.shape[1]
    y = []
    pre = []

    # 反转换类别的独热编码
    for i in range(cols):
        if y_test[0][i] == 0 and y_test[1][i] == 0 and y_test[2][i] == 1:
            y.append('setosa')
        elif y_test[0][i] == 0 and y_test[1][i] == 1 and y_test[2][i] == 0:
            y.append('versicolor')
        elif y_test[0][i] == 1 and y_test[1][i] == 0 and y_test[2][i] == 0:
            y.append('virginica')

    for j in range(cols):
        if result[0][j] == 0 and result[1][j] == 0 and result[2][j] == 1:
            pre.append('setosa')
        elif result[0][j] == 0 and result[1][j] == 1 and result[2][j] == 0:
            pre.append('versicolor')
        elif result[0][j] == 1 and result[1][j] == 0 and result[2][j] == 0:
            pre.append('virginica')
        else:
            pre.append('unknown')

    # 将特征和类别矩阵拼接起来
    real = np.column_stack((x_test.T, y))
    prediction = np.column_stack((x_test.T, pre))

    # 转换成DataFrame类型，并添加columns
    df_real = pd.DataFrame(real, index=None, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species'])
    df_prediction = pd.DataFrame(prediction, index=None, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species'])

    # 将特征列转换为float类型，否则radviz会报错
    df_real[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']] = df_real[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']].astype(float)
    df_prediction[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']] = df_prediction[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']].astype(float)

    # 绘图
    plt.figure('真实分类')
    pd.plotting.radviz(df_real, 'Species', color=['blue', 'green', 'red', 'yellow'])
    plt.figure('预测分类')
    pd.plotting.radviz(df_prediction, 'Species', color=['blue', 'green', 'red', 'yellow'])
    plt.show()



def testDemo():
    dataSet = readingDatas()
    trainData,testData = randomData(dataSet, 0.8)
    trainData = pd.DataFrame(trainData)
    testData = pd.DataFrame(testData)
    # 读取数据
    data_set = trainData

    # 取数据方法：
    X = data_set.iloc[:, 0:4].values.T          # 前四列是特征，T表示转置
    Y = data_set.iloc[:, 4:].values.T           # 后三列是标签

    Y = Y.astype('uint8')

    # 开始训练
    start_time = datetime.datetime.now()
    # 输入4个节点，隐层10个节点，输出3个节点，迭代10000次
    parameters = nn_model(X, Y, n_h=10, n_input=4, n_output=3, num_iterations=10000, print_cost=True)
    end_time = datetime.datetime.now()
    print("用时：" + str((end_time - start_time).seconds) + 's' + str(round((end_time - start_time).microseconds / 1000)) + 'ms')

    # 对模型进行测试
    data_test = testData
    x_test = data_test.iloc[:, 0:4].values.T
    y_test = data_test.iloc[:, 4:].values.T
    y_test = y_test.astype('uint8')

    result = predict(parameters, x_test, y_test)

    # 分类结果可视化
    result_visualization(x_test, y_test, result)

if __name__ == "__main__":
    testDemo()
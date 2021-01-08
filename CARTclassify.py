

from random import seed  # 初始化随机数发生器的种子值
from random import randrange  # 指定区间随机不重复抽取整数
from csv import reader  # 读取CSV文件
import numpy as np #矩阵运算库
import pandas as pd #提供高性能易用数据类型和分析工具
from scipy.io import arff #方便导入arff文件数据

'''
该算法为CART分类算法

数据1来源：diabetes.arff 糖尿病人的各项数据
数据内容包括：preg（怀孕次数）、plas（葡萄糖浓度）、pres（血压）、skin（皮肤厚度）、insu（胰岛素）、mass（体重）、 pedi（谱系功能）、 age（年龄）、 class(是否为糖尿病人)
前8列为糖尿病人特征特征、最后一列为是否患有糖尿病

数据2来源：data_banknote_authentication.csv(钞票数据集)
这是从纸币鉴别过程中的图像里提取的数据，用来预测钞票的真伪的数据集
第一列：图像经小波变换后的方差(variance)(连续值)；
第二列：图像经小波变换后的偏态(skewness)(连续值)；
第三列：图像经小波变换后的峰度(curtosis)(连续值)；
第四列：图像的熵(entropy)(连续值)；
第五列：钞票所属的类别(整数，0或1)
'''


def readingDatas():
    '''
    读入数据，并修改数据，添加一列数据且值恒为1，同时将最后一列枚举数据转换为0、1
    :return:
    '''
    array = arff.loadarff("./Dataset/diabetes.arff")
    df = pd.DataFrame(array[0])
    # df.insert(0, 'constant', 1)  # 添加constant列 值恒为1
    df = df.replace(b'tested_negative', 0)
    df = df.replace(b'tested_positive', 1)
    data_array = np.array(df)
    dataSet = data_array.tolist()
    return dataSet



def load_csv(filename):
    '''
    读取CSV文件返回整体数据集列表
    Parameters
    ----------
    filename : String
        路径名称
    Returns
    -------
    dataset : list只含有效行的数据集列表.
    '''
    dataset = list()  # 空列表
    with open(filename, 'r') as file:  # 只读 返回文件对象
        csv_reader = reader(file)  # reader(open(filename, 'r'))#所有行
        for row in csv_reader:
            if not row:  # 去除CSV文件空行
                continue
            dataset.append(row)
    return dataset



def str_column_to_float(dataset, column):
    '''
    数值转换将string转换为float
    :param dataset:
    :param column:
    :return:
    '''

    for row in dataset:
        row[column] = float(row[column].strip())  # 提取字符串值（去掉字符串周围的空白和''）强制转换为float



def cross_validation_split(dataset, n_folds):
    '''
    将数据集转换为k组，采用k折交叉熵
    :param dataset:
    :param n_folds:
    :return:
    '''
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)  # 每一折样本数量
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))  # 0到n-1
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split



def accuracy_metric(actual, predicted):
    '''
    :param actual: 真实标签.
    :param predicted: 预测标签
    :return:
    '''

    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    '''
    使用交叉验证拆分评估算法
    :param dataset: 所有已知标签的样本集.
    :param algorithm: 参与评价的算法.
    :param n_folds: 折数.
    :param args: .取决于算法的额外参数
    :return:   scores : TYPE
        DESCRIPTION.
    '''

    folds = cross_validation_split(dataset, n_folds)  #将数据集均分成n等分，列表方式返回。N个版本的训练集与测试集

    scores = list()

    for index,fold in enumerate(folds):
        train_set = list(folds)
        train_set.remove(fold)  # 去掉本折
        train_set = sum(train_set, [])  # 将[[],[],[],[]]变为[]

        test_set = list()
        for row in fold:  # 将本折作为测试集
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None  # 去掉正确的类别标号

        predicted = algorithm(train_set, test_set, *args)  # 接收的关于测试集的预测标签

        actual = [row[-1] for row in fold]  # 真实的预测标签
        accuracy = accuracy_metric(actual, predicted)
        print('accuracy',index+1,':',accuracy)
        scores.append(accuracy)
    return scores



def test_split(index, value, dataset):
    '''
    根据属性和属性值拆分数据集
    :param index: int
        切分特征的序号.
    :param value: int
        切分特征对应的切分阈值.
    :param dataset: list
        到达当前节点的训练样本集.
    :return:     left : list
        左子集.
    right : list
        右子集.
    '''

    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)  # scikit中《=
        else:
            right.append(row)
    return left, right



def gini_index(groups, classes):
    '''
    计算拆分数据集的基尼系数
        CART 二叉树 只有左右子集.
    针对切分特征中的切分点划分后的子集计算加权基尼指数
    :param groups:  list
        左右两个子集.
    :param classes: TYPE
        所有类别标号.
    :return:     gini : float
        基尼指数.
    '''

    # 总的样本数--count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))  # 求和[[左子集数量],[右子集数量]]得总的样本集数量，权重的分母

    # 划分后的加权GINI指数--sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))  # 获取当前子集的样本数量

        # avoid divide by zero#没有进行排序选阈值
        if size == 0:  # 阈值为 最小值，最大值
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size  # [所有类别标号].count()-->class_val的数量 作为分子
            score += p * p  # 平方和

        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)  # 不纯度 * 权重

    return gini



def get_split(dataset):
    '''
    为数据集选择最佳分割点
        利用 1-6 划分左右子集，1-7 计算基尼指数进行划分,进行切分特征和切分阈值的优选
    :param dataset: list
        到达该节点的样本集.
    :return:     dict
        {特征序号，特征切分阈值，[左右子集]}.
    '''

    class_values = list(set(row[-1] for row in dataset))  # [到达该节点的类别标号]

    # 初始化最小基尼指数对应的特征序号，切分点，基尼指数值，分组列表
    b_index, b_value, b_score, b_groups = 999, 999, 999, None

    # 针对每个备选的特征
    for index in range(len(dataset[0]) - 1):  # [1,2,3,4,标号]

        # 考查每个备选的切分阈值
        for row in dataset:
            groups = test_split(index, row[index], dataset)  # 为了简单快速生成树，未sort,只是选了每个特征作为阈值。#生成左右两个子集
            gini = gini_index(groups, class_values)
            if gini < b_score:  # 因为找的是最小基尼指数的点.
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups

    # 在当前结点处，存放字典形式的信息：切分特征、切分点、划分的两个子集
    return {'index': b_index, 'value': b_value, 'groups': b_groups}



def to_terminal(group):
    '''
    创建终端节点值,    基于到达当前节点的训练集，生成该节点的预测值
    :param group:     基于到达当前节点的训练集，生成该节点的预测值
    :return:     预测类别
        数量最多的类别.如果一样多，返回标号最小的类别.
    '''

    outcomes = [row[-1] for row in group]  # [类别标号]
    return max(set(outcomes), key=outcomes.count)  # 返回出现次数最多的类别。


def split(node, max_depth, min_size, depth):
    '''
    为节点创建子拆分或建立终端
    基于当前节点生成当前结点的后续子节点,或直接作为叶子结点生成预测结果（节点信息，3个限制条件）要么递归划分，要么直接作为叶子终结

    :param node: dict
        当前结点的字典形式的描述.{'index':b_index, 'value':b_value, 'groups':b_groups}
    :param max_depth: TYPE
        最大深度.
    :param min_size:   TYPE
        最小样本数
    :param depth: TYPE
        当前深度.
    :return:
    '''

    # (1)获取该结点的左右子集，为后续更新结点信息更新做准备
    left, right = node['groups']
    del (node['groups'])

    # (2)判断当前结点是否直接作为叶子结点--check for a no split
    # 如果是，不再分裂，直接返回
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        # print('当前节点为叶子节点')
        return

    # (2)check for max depth 若已为最大深度，后续两个子结点直接作为叶子结点
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)  # 得到预测类别。
        # print('创建两个叶子节点')
        return

    # (3)process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
        # print('左节点为叶子节点')
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
        # print('创建左节点')

    # (4)process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
        # print('右节点为叶子节点')
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)
        # print('创建右节点')



def build_tree(train, max_depth, min_size):
    '''
    简历决策树
    :param train: list
        初始用于生成树模型的训练样本集.
    :param max_depth: int
        最大深度.
    :param min_size: int
        最小样本数.
    :return: list
        根节点.
    '''
    root = get_split(train)  # 1-8
    split(root, max_depth, min_size, 1)  # 递归调用生成多重嵌入字典.
    return root  # 多重字典


def predict(node, row):
    '''
    用决策树进行预测
        单极的测试，基于该node节点的特征和阈值对样本进行测试
    :param node: dict 或者其他
        当前节点.
    :param row: list
        样本.
    :return:     TYPE
        node（dict或者预测输出）.
    '''

    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):  # 用字典方式表示
            return predict(node['left'], row)
        else:
            return node['left']  # 见1-10
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def decision_tree(train, test, max_depth, min_size):
    '''
    构建决策树，充当algorithm
    :param train: list
        训练集.
    :param test: list
        测试集.
    :param max_depth: int
        最大深度.
    :param min_size: int
        最小样本数.
    :return:     关于测试集的预测结果.
    '''

    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return (predictions)  # 关于测试集的预测结果






if __name__ == '__main__':
    seed(1)

    #5折交叉验证
    n_folds = 5
    max_depth = 5
    min_size = 20


    dataSet = readingDatas()


    print('diabetes.arff数据集：')
    scores2 = evaluate_algorithm(dataSet, decision_tree, n_folds, max_depth, min_size)
    print('Scores: %s' % scores2)
    print('平均准确率为: %.3f%%' % (sum(scores2) / float(len(scores2))))


    # 输入数据
    filename = './Dataset/data_banknote_authentication.csv'
    dataset = load_csv(filename)

    #string转换为 float
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)


    print('data_banknote_authentication数据集：')
    scores1 = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
    print('Scores: %s' % scores1)
    print('平均准确率为: %.3f%%' % (sum(scores1) / float(len(scores1))))






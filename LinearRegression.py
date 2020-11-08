import math  #数学基本运算
import matplotlib.pyplot as plt #图形显示
import random  #随机数
import numpy as np #矩阵运算库
import pandas as pd #提供高性能易用数据类型和分析工具
import seaborn as sns #绘制数据分布，数据观察函数
from scipy.io import arff #方便导入arff文件数据
import sys #用于表示最大值和最小值

'''
该算法为线性回归算法
数据来源：abalone.data 鲍鱼的各项数据
数据内容包括：性别、长度、直径、高度、重量、肉重量、内脏重量、外壳重量、年龄
前8列为鲍鱼特征、最后一列为鲍鱼年龄（预测结果）
'''

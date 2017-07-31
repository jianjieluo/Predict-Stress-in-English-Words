"""
本模块用来集成测试不同的模型的预测结果
"""

import helper
from submission import *
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split 
from sklearn.metrics import f1_score

from sklearn import tree
import sklearn.naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors


if __name__ == '__main__':
    # 先把训练集读进来，然后弄出特征矩阵和对应的label数组
    # 然后用train_test_split来在训练集里面分出训练集和测试集，（这个过程只用到特征矩阵和label数组，与原来的数据字符串没有关系了）
    raw_data = helper.read_data('./asset/training_data.txt')

    # 在这里提取出想要的特征矩阵features和对应的分类数组labels数组
    features, labels = training_preprocess(raw_data)

    clf1 = tree.DecisionTreeClassifier(criterion='gini')
    clf2 = tree.DecisionTreeClassifier(criterion='entropy')
    clf3 = sklearn.naive_bayes.GaussianNB()
    clf4 = sklearn.naive_bayes.BernoulliNB()
    clf5 = LogisticRegression()
    clf6 = neighbors.KNeighborsClassifier(algorithm='kd_tree')

    clfs = [('gini_dtree', clf1), ('entr_dtree',clf2), ('GaussianNB', clf3), ('BernoulliNB', clf4), ('LogisticRegression',clf5), ('kd_tree', clf6)]

    res = []
    for clf in clfs:
        train_f1_li = []
        test_f1_li = []
        for _ in range(300):
            x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2)
            clf[1].fit(x_train, y_train)

            answer = clf[1].predict(x_train)
            answer2 = clf[1].predict(x_test)

            train_f1_li.append(f1_score(y_train, answer, average='micro'))
            test_f1_li.append(f1_score(y_test, answer2, average='micro'))

        res.append([clf[0], np.mean(train_f1_li), np.mean(test_f1_li)])
    
    matrix = np.array(res).transpose()

    df = pd.DataFrame({
                   'mean train f1' : matrix[1],
                   'mean test f1' : matrix[2],
    }, index = matrix[0])
    
    print (df)
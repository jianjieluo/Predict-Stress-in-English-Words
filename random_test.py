"""Random test trainig and testing module

This module use cross_validation to check the f1 of our classifier
in order to get a better classifier to the predict problem.

"""

import helper
from submission import *
import numpy as np
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import f1_score


if __name__ == '__main__':
    # 先把训练集读进来，然后弄出特征矩阵和对应的label数组
    # 然后用train_test_split来在训练集里面分出训练集和测试集，（这个过程只用到特征矩阵和label数组，与原来的数据字符串没有关系了）
    raw_data = helper.read_data('./asset/training_data.txt')

    # 在这里提取出想要的特征矩阵features和对应的分类数组labels数组
    features, labels = training_preprocess(raw_data)

    clf = get_selected_classifier()


    # 模式选择，输入y表示使用均值模拟，多次交叉检验，减少误差
    # 输入其他表示不使用均值模拟，减少运行时间
    print("Do you want to test for multiple times? y/n [default:n]")
    # choice = input()
    choice = 'y'

    if choice != 'y':
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2)
        clf.fit(x_train, y_train)
        answer = clf.predict(x_train).round()
        answer2 = clf.predict(x_test).round()
        print('f1 for train = ' , f1_score(y_train, answer, average='micro'))
        print('f1 for test = ' , f1_score(y_test, answer2, average='micro'))

    else:
        # 进行多次随机测试，并且求出均值，以此来减少偶然型
        train_f1_li = []
        test_f1_li = []
        for i in range(300):
            print ('time ', i)

            x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2)
            clf.fit(x_train, y_train)

            # 只输出一次importance，减少运行时间
            if i == 1:
                print ('features importance: ', clf.feature_importances_)

            answer = clf.predict(x_train)
            answer2 = clf.predict(x_test)

            train_f1_li.append(f1_score(y_train, answer, average='micro'))
            test_f1_li.append(f1_score(y_test, answer2, average='micro'))

        print('mean f1 for train = ' , np.mean(train_f1_li))
        print('mean f1 for test = ' , np.mean(test_f1_li))
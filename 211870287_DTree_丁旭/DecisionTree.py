from sklearn.datasets import load_iris
import random
from math import log

from sklearn.metrics import f1_score

from drawTree import createPlot

import operator

"""
初始化数据集,返回一个列表,值为一个样本的属性与其标签,比如一个样本为[1.5, 2.5, 3.5, 4.5, 2]
# 则前四个元素为对应的属性,最后一个元素2为这个样本所属的类别
"""


def init_data_set():
    iris = load_iris()  # 导入数据集iris
    iris_feature = iris.data.tolist()  # 样本属性
    print('iris_feature: ',iris_feature)
    iris_target = iris.target.tolist()  # 样本类别
    print('iris_category: ',iris_target)
    for i in range(len(iris_feature)):
        iris_feature[i].append(iris_target[i])
    return iris_feature


"""
划分数据集 将total_data_set 中比例为 split_rate的部分划分为训练集 剩下的作为数据集
"""


def create_train_and_test_set(total_data_set, split_rate=0.75):
    # 0的是测试集，1的是训练集
    length = len(total_data_set)
    train_num = int(length * split_rate)
    test_num = length - train_num

    random_list = [1 for _ in range(train_num)] + [0 for _ in range(test_num)]
    random.shuffle(random_list)
    test_set = []
    train_set = []
    for i in range(length):
        if random_list[i] == 0:
            test_set.append(total_data_set[i])
        else:
            train_set.append(total_data_set[i])
    return test_set, train_set


"""
计算给定的数据集的信息熵
"""


# 计算信息熵
def calculate_Entropy(data_set):
    label = {}
    for i in data_set:
        label[i[-1]] = label.get(i[-1], 0) + 1
    entropy = 0.0
    for i in label:
        pi = float(label[i]) / len(data_set)
        entropy -= pi * log(pi, 2)
    return entropy


"""
根据pos和value的值划分成两个数据集
对于每个样本 按其第pos个属性的值与value的值的大小关系为其分类
"""


def split_Set(data_set, pos, value):
    less_Set = []
    more_Set = []
    for item in data_set:
        if item[pos] < value:
            less_Set.append(item)
        else:
            more_Set.append(item)
    return less_Set, more_Set


"""
选择最好的特征值进行分类
返回最好的特征值以及最佳的信息增益
例如返回[2,3.0] 与 0.8
则代表最好的分类特征为第2个属性(下标从0开始),并应该以3.0为划分标准将其划分为两部分,其所能达到的信息增益为0.8
"""


def choose_best_split(data_set):
    base_Ent = calculate_Entropy(data_set)
    best_increase = 0.0
    best_feature = [-1, -1]
    for i in range(4):
        features = [j[i] for j in data_set]
        unique = set(features)
        for feature in unique:
            less_Set, more_Set = split_Set(data_set, i, feature)
            tmp = len(less_Set) / float(len(data_set))
            new_Ent = tmp * calculate_Entropy(less_Set) + (1 - tmp) * calculate_Entropy(more_Set)
            increase = base_Ent - new_Ent

            if increase > best_increase:
                best_increase = increase
                best_feature = [i, feature]
    return best_feature, best_increase


"""
使用递归的方式以字典的形式构造并返回决策树
叶子结点只有一个键为class,值为其应属于的类别
否则在node键中存储分类的特征 并在left和right键中存储左子树和右子树构成的字典
"""


def create_tree(data_set):
    myTree = {}
    label = [i[-1] for i in data_set]
    label_set = set(label)
    if len(label_set) == 1:
        myTree['class'] = label[0]
        return myTree
    best_feature, best_increase = choose_best_split(data_set)

    # 进行预剪枝
    if best_increase == 0.0:
        # myTree['class'] = max(label, key=label.count)
        myTree['class'] = majorityCnt(label)
        return myTree

    myTree['node'] = best_feature
    less_Set, more_Set = split_Set(data_set, best_feature[0], best_feature[1])
    myTree['left'] = create_tree(less_Set)
    myTree['right'] = create_tree(more_Set)
    return myTree


def majorityCnt(classList):
    classCount = {}

    # 统计每个类标签出现的频率
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    # 利用operator操作键值排序字典
    sortedClassCount = sorted(classCount.items(),
        key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]  # 返回classList中出现次数最多的元素


"""
算法同 create_tree方法
但是返回的字典更易于绘制可视化决策树图
"""


def draw_tree(data_set):
    myTree = {}
    label = [i[-1] for i in data_set]
    label_set = set(label)
    if len(label_set) == 1:
        return 'type:' + str(label[0]) + '\nsample:' + str(len(data_set))
    best_feature, best_increase = choose_best_split(data_set)


    string = 'X[' + str(best_feature[0]) + ']<' + str(best_feature[1])
    string += '\nbest_increase=' + str(round(best_increase, 3))
    string += '\nsample:' + str(len(data_set))
    myTree[string] = {}
    less_Set, more_Set = split_Set(data_set, best_feature[0], best_feature[1])
    myTree[string]['True'] = draw_tree(less_Set)
    myTree[string]['False'] = draw_tree(more_Set)
    return myTree



def draw_pre_tree(data_set):
    myTree = {}
    label = [i[-1] for i in data_set]
    label_set = set(label)
    if len(label_set) == 1:
        return 'type:' + str(label[0]) + '\nsample:' + str(len(data_set))
    best_feature, best_increase = choose_best_split(data_set)

    # 进行预剪枝
    if best_increase < 0.15:
        # myTree['class'] = max(label, key=label.count)
        # myTree['class'] = majorityCnt(label)
        return 'type:' + str(majorityCnt(label)) + '\nsample:' + str(len(data_set))

    string = 'X[' + str(best_feature[0]) + ']<' + str(best_feature[1])
    string += '\nbest_increase=' + str(round(best_increase, 3))
    string += '\nsample:' + str(len(data_set))
    myTree[string] = {}
    less_Set, more_Set = split_Set(data_set, best_feature[0], best_feature[1])
    myTree[string]['True'] = draw_tree(less_Set)
    myTree[string]['False'] = draw_tree(more_Set)
    return myTree
"""
对输入的样本input 返回其应该属于的类别
"""


def predict(tree, input):
    if 'class' in tree:
        return tree['class']
    feature = tree['node']
    if input[feature[0]] < feature[1]:
        return predict(tree['left'], input)
    else:
        return predict(tree['right'], input)


# 分类
def classify(tree, test_set):
    correct_count = 0
    for item in test_set:
        if predict(tree, item[:-1]) == item[-1]:
            correct_count += 1
    accuracy = float(correct_count) / len(test_set)
    return accuracy



if __name__ == '__main__':
    data_set = init_data_set()
    # print(data_set)
    test_set, train_set = create_train_and_test_set(data_set, 0.8)
    res = create_tree(train_set)
    y_true = [example[-1] for example in test_set]
    y_pred = [predict(res, example) for example in test_set]
    # # 用Micro-F1和Macro-F1分数进行验证集评估
    print('micro-F1分数为:', f1_score(y_true, y_pred, labels=[0, 1, 2], average='micro'))
    print('macro-F1分数为:', f1_score(y_true, y_pred, labels=[0, 1, 2], average='macro'))
    # # 绘制可视化结果
    # createPlot(draw_tree(train_set))
    # createPlot(draw_pre_tree(train_set))

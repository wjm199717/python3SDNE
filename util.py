import numpy as np
#sklearn模块是机器学习模块，model_selection模块主要是对训练集数据进行交叉验证划分，
#然后该模块也可以用来划分训练集和测试集，还可以调整模型参数，模型评估，学习/验证曲线
from sklearn.model_selection import train_test_split
#linear_model该模块存储了一些封装好的线性模型，线性模型本质就是最小二乘模型，只是不同的模型有不同的损失函数
#linear_model里面的模型都是为了拟合最小二乘模型
from sklearn.linear_model import LogisticRegression
#metrics模块存储的是度量模型好坏的方法
from sklearn.metrics import f1_score
#multiclass模块存储的是已经封装好的各种分类器
from sklearn.multiclass import OneVsRestClassifier
#pdb是python自带的一个包，为python程序提供了一种交互的源代码调试功能
#import pdb

class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def getSimilarity(result):
    print ("getting similarity...")
    #dot()函数处理的是矩阵间的乘积或者向量间的内积
    #a.dot(b) 与 np.dot(a,b)效果相同。
    return np.dot(result, result.T)

#check_index是个列表
def check_reconstruction(embedding, graph_data, check_index):
    #precision用在分类问题上，好像是二分类问题
    def get_precisionK(embedding, data, max_index):
        #精确率就是你预测为正样本当中，有多少预测正确了
        print ("get precisionK...")
#将embedding自动变为1维的，-1算是一个占位符，具体有多少个元素自动计算，-1自动计算的意思
        similarity = getSimilarity(embedding).reshape(-1)
#argsort()函数是将参数列表中的元素从小到大排列，提取其对应的index(索引)，然后输出到一个列表
        sortedInd = np.argsort(similarity)
        cur = 0
        count = 0
        precisionK = []
#b = a[i:j:s]表示：i,j与切片一样，但s表示步进，缺省为1.a[i:j:1]相当于a[i:j]
#当s<0时，i缺省时，默认为-1. j缺省时，默认为-len(a)-1
#所以a[::-1]相当于 a[-1:-len(a)-1:-1]，也就是从最后一个元素到第一个元素复制一遍，即倒序。        
        sortedInd = sortedInd[::-1]
#通过节点的embedding计算节点间的相似性，然后取相似性最大的前k个，看看这k个里面有多少是实际存在边的
#precisionk就是假定预测前k个是正类，其余的是负类，计算预测的前k个里面有多少被正确判别为正类
        for ind in sortedInd:
            #sortedInd存储的是节点间的相似性
            #x,y存储的是该相似性在邻接矩阵中的位置
            x = int(ind / data.N)
            y = int(ind % data.N)
            count += 1
            if (data.adjmatrix[x][y] == 1 or x == y):
                cur += 1 
            precisionK.append(1.0 * cur / count)
            if count > max_index:
                #退出循环，这里的退出循环是不再执行循环了，并不是退出这一次循环
                break
        return precisionK
    #np.max(a)取a列表中的元素最大值
    precisionK = get_precisionK(embedding, graph_data, np.max(check_index))
    ret = []
    for index in check_index:
        print ("precisonK[%d] %.2f" % (index, precisionK[index - 1]))
        ret.append(precisionK[index - 1])
    return ret

def check_link_prediction(embedding, train_graph_data, origin_graph_data, check_index):
    def get_precisionK(embedding, train_graph_data, origin_graph_data, max_index):
        print ("get precisionK...")
        similarity = getSimilarity(embedding).reshape(-1)
        #sortedInd该列表存储的是similarity列表里面元素的索引
        sortedInd = np.argsort(similarity)
        cur = 0
        count = 0
        precisionK = [] 
        #sortedInd[::-1]相当于逆序切片一个列表的全部元素
        sortedInd = sortedInd[::-1]
        N = train_graph_data.N
        for ind in sortedInd:
            x = ind / N
            y = ind % N
            if (x == y or train_graph_data.adj_matrix[x][y] == 1):
                continue 
            count += 1
            if (origin_graph_data.adj_matrix[x][y] == 1):
                cur += 1
            precisionK.append(1.0 * cur / count)
            if count > max_index:
                break
        return precisionK
    precisionK = get_precisionK(embedding, train_graph_data, origin_graph_data, np.max(check_index))
    ret = []
    for index in check_index:
        print ("precisonK[%d] %.2f" % (index, precisionK[index - 1]))
        ret.append(precisionK[index - 1])
    return ret
 

def check_multi_label_classification(X, Y, test_ratio = 0.9):
    def small_trick(y_test, y_pred):
        y_pred_new = np.zeros(y_pred.shape,np.bool)
        #np.flip对矩阵进行翻转，具体怎么翻转查百度
        sort_index = np.flip(np.argsort(y_pred, axis = 1), 1)
        for i in range(y_test.shape[0]):
            num = sum(y_test[i])
            for j in range(num):
                y_pred_new[i][sort_index[i][j]] = True
        return y_pred_new
    #x代表的是embedding，将数据集划分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = test_ratio)
    clf = OneVsRestClassifier(LogisticRegression())
    #用训练集来训练这个逻辑回归分类器，二分类
    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(x_test)
    
    ## small trick : we assume that we know how many label to predict
    y_pred = small_trick(y_test, y_pred)
   
#衡量模型的两个度量指标，值越大说明模型越好，该指标工作原理已经被封装好了，直接用就行了
    micro = f1_score(y_test, y_pred, average = "micro")
    macro = f1_score(y_test, y_pred, average = "macro")
    return ("micro_f1: %.4f macro_f1 : %.4f" % (micro, macro))
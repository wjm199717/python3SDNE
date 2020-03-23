import scipy.io as sio
import numpy as np
import util 
import random
#import copy
#python中scipy模块中，有一个模块叫sparse模块，就是专门为了解决稀疏矩阵而生。
#sparse中有7种矩阵类型，这个模块是用来对稀疏矩阵进行压缩，减少内存占用
#csr_matrix，全名为Compressed Sparse Row，是按行对矩阵进行压缩的。
#csr_matrix适合用来进行矩阵的运算
#from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix

class Graph(object):
    def __init__(self, file_path, ng_sample_ratio):
        #以.为分割符保留列表的最后一个元素，这里元素就是txt
        suffix = file_path.split('.')[-1]
        self.st = 0
        self.is_epoch_end = False
        if suffix == "txt":
            fin = open(file_path, "r")
            #文件的每一行都有一个换行符/n
            #读取文件的每一行，将每一行都存在列表中
            #读取文件内容时，读取的都是字符串
            #strip()是消除字符串的空格或者换行符
            #split()默认是以空格符或/n为分割符，返回的是一个列表
            #readline()读取的是一行，读取完后返回一个字符串
            firstLine = fin.readline().strip().split()
            #整个数据集的总节点数
            self.N = int(firstLine[0])
            #整个数据集的总边数
            self.E = int(firstLine[1])
            self.__is_epoch_end = False
            
            #自己添加的初始化self.link的代码
            self.links = []
            for line in fin.readlines():
                line = line.strip().split()
                line.append(1)
                self.links.append(line)
            del self.links[0]
            
            
#用dok_matrix创建了一个矩阵，适用的场景是逐渐添加矩阵的元素
#dok_matrix()中元组存放的是矩阵的大小，后面存放的是矩阵中每个元素的类型
#dok_matrix的策略是采用字典来记录矩阵中不为0的元素。
# 自然，字典的key存的是记录元素的位置信息的元组，
#  value是记录元素的具体值。
#dok_matrix返回的是一个字典,字典中的键是一个元组存储的是矩阵非零元素的位置，
#字典中的值是一个数字，存储的是非零元素的值，由于dok_matrix返回的是一个字典，
#我们可以用toarray()转化成2维数组，数组跟列表是不一样的，数组之间是没有逗号的
            self.adj_matrix = dok_matrix((self.N, self.N), np.int_)
            count = 0
            #readlines()返回的是一个列表
            #通过训练数据文件构成邻接矩阵，训练的图结构就已经形成了
            for line in fin.readlines():
                line = line.strip().split()
                x = int(line[0])
                y = int(line[1])
                
                #自己添加的代码
                if x == 10312 and y ==333983:
                    continue
                
                #遍历数据集文件然后构建邻接矩阵
                self.adj_matrix[x, y] = 1
                self.adj_matrix[y, x] = 1
                #count计算有多少个边
                count += 1
            fin.close()
#            tocsr()返回一个该矩阵的压缩稀疏行格式，返回3个数组构成的列表
#返回稀疏矩阵的csr_matrix形式,稀疏矩阵是adj_matrix
            #以压缩后的形式存储邻接矩阵，减少内存消耗
            self.adjmatrix = list(self.adj_matrix.toarray())
            self.adj_matrix = self.adj_matrix.tocsr()
        else:
            try:
                #读取mat文件，mat文件中的邻接矩阵是以字典的形式存储的
                self.adj_matrix = sio.loadmat(file_path)["graph_sparse"].tocsr()
            except:
                self.adj_matrix = sio.loadmat(file_path)["traingraph_sparse"].tocsr()
            self.N, _ = self.adj_matrix.get_shape()
            self.E = self.adj_matrix.count_nonzero() / 2
        if (ng_sample_ratio > 0):
            self.__negativeSample(int(ng_sample_ratio*self.E))
            #np.arange()函数返回一个有终点和起点的固定步长的排列
            #这里是从0至self.N-1，差一原则
        self.order = np.arange(self.N)
        print ("Vertexes : %d  Edges : %d ngSampleRatio: %f" % (self.N, self.E, ng_sample_ratio))
        
    def __negativeSample(self, ngSample):
        #采集负样本，也就是两个节点之间没有边的样本，2个节点算一个负样本
        print ("negative Sampling")
        size = 0
        while (size < ngSample):
#一共有self.N个节点，但是我们编号是从0开始编号，所有一直编到self.N-1
            xx = random.randint(0, self.N-1)
            yy = random.randint(0, self.N-1)
            if (xx == yy or self.adj_matrix[xx][yy] != 0):
                continue
            #邻接矩阵中-1就代表是个负样本
            self.adj_matrix[xx][yy] = -1
            self.adj_matrix[yy][xx] = -1
            size += 1
        print ("negative Sampling done")
        
    def load_label_data(self, filename):
        with open(filename,"r") as fin:
            firstLine = fin.readline().strip().split()
#firstLine[1]代表的是标签数，搭建了一个节点数为行数，标签数为列数的矩阵
#python中定义类，初始化方法中并不是需要将类中所有定义的属性都初始化，
            #类中的方法也可以定义新的属性，只要给该属性赋值了就行
            #firstLine[1]存储的是一共有多少种标签
            self.label = np.zeros([self.N, int(firstLine[1])], np.bool)
#readlines()返回一个字符串列表，从文件中读到的都是字符串
            lines = fin.readlines()
            for line in lines:
                line = line.strip().split(' : ')
                if len(line) > 1:
                    labels = line[1].split()
                    for label in labels:
                        #构成标签矩阵
                        self.label[int(line[0])][int(label)] = True

    
    def sample(self, batch_size, do_shuffle = True, with_label = False):
        if self.is_epoch_end:
            if do_shuffle:
                #打乱self.order列表的顺序
                #这里的self.order[0:self.N]是一个切片，切片也是差一原则
                np.random.shuffle(self.order[0:self.N])
            else:
                self.order = np.sort(self.order)
            self.st = 0
            self.is_epoch_end = False 
        mini_batch = util.Dotdict()
        en = min(self.N, self.st + batch_size)
        #采一个批次的节点
        index = self.order[self.st:en]     
        #一个矩阵行数是index列表的长度，列数还是原来邻接矩阵的列数
        #mini_batch是一个对象
        mini_batch.X = self.adj_matrix[index].toarray()
        #将上面那个矩阵的列数给切掉一部分，使得行数跟列数相同
        #第一个[:]是将原2维数组复制一遍得到一个副本，然后在复制后的这个新数组上进行切片操作
        #切片操作的2维数组，第一维全部取，第二维从0开始取到index-1
        mini_batch.adjacent_matriX = self.adj_matrix[index].toarray()[:][:,index]
        #这里是取标签矩阵
        if with_label:
            mini_batch.label = self.label[index]
#在数据集中一个batch一个batch的顺着往下取，每次取一个batch的数据，直到整个数据集取完结束
        if (en == self.N):
            en = 0
            self.is_epoch_end = True
        self.st = en
        #返回的是一个实例
        return mini_batch
    
    def subgraph(self, method, sample_ratio):
        new_N = int(sample_ratio * self.N)
        cur_N = 0
        if method == 'link':
            #以边的方式构建子图
            #从有连边的图中随机抽取一部分连边，构成原来图的一个子图
            new_links = []
            self.adj_matrix = np.zeros([self.N, self.N], np.int_)
            self.order = {}
            while (cur_N < new_N):
            #random.random()用于生成一个0到1的随机浮点数
                p = int(random.random() * self.E)
            #self.links可能是一个[[a,b],[c,d],[e,f]],一共有self.E个元素
            #link可能是一个两个元素的列表[a, b]
            #links可能是一个元素为列表的列表[[a, b],[c, d],[e, f]]
                link = self.links[p]
                if self.adj_matrix[link[0]][link[1]] == 0:
                    new_links.append(link)
                    self.adj_matrix[link[0]][link[1]] = 1
                    self.adj_matrix[link[1]][link[0]] = 1
                    if link[0] not in self.order:
                        #将子图的节点存储到self.order属性里面
                        self.order[link[0]] = 1
                        cur_N += 1
                    if link[1] not in self.order:
                        self.order[link[1]] = 1
                        cur_N += 1
            self.links = new_links
            self.order = self.order.keys()
            self.N = new_N
            print (len(self.links))
            #return self返回的是一个类的实例化对象，self指的就是实例化对象
            return self
        elif method == "node":
            #以节点的方式构建子图
            self.adj_matrix = np.zeros([self.N, self.N], np.int_)
            self.order = {}
            new_links = []
            while (cur_N < new_N):
                #random.random()返回随机生成的一个实数，它在[0,1)范围内。
                #随机抽取节点，不能重复抽，一直抽到满足条件为止
     #抽完节点完毕以后，再去找这些节点的连边，连边与原始图的连边保持一样
                p = int(random.random() * self.N)
                if p not in self.order:
                    self.order[p] = 1
                    cur_N += 1
            for link in self.links:
                if link[0] in self.order and link[1] in self.order:
                    self.adj_matrix[link[0]][link[1]] = 1
                    self.adj_matrix[link[1]][link[0]] = 1
                    new_links.append(link)
            self.order = self.order.keys()
            self.N = new_N
            self.links = new_links
            print (len(self.links))
            return self
            pass
        elif method == "explore": 
            new_adj_matrix = np.zeros([self.N, self.N], np.int_)
            self.order = {}
            new_links = []
            while (cur_N < new_N):
                #random.random()返回一个[0,1)区间的随机数
                #p是一个随机节点
                p = int(random.random() * self.N)
                k = int(random.random() * 100)
#以p节点为起点，再随机选取一个邻居节点，再以邻居节点为起点，再随机选择一个邻居节点
#然后一直到循环次数结束，这就相当于一次以p为起点的，长度为k的随机游走，执行new_N次随机游走
                for i in range(k):
                    #p是键名，是被选中的节点
                    if p not in self.order:
                        self.order[p] = 1
                        cur_N += 1
                    #nonzero()返回两个数组，一个数组是记录非零元素所在的行
                    #一个数组记录非零元素所在的列
                    #取两个数组对应位置的元素，就是邻接矩阵中一个非零元素的位置
                    #所以这里b是一个2维数组
                    #在邻接矩阵中取p节点所在的这一行中的非零元素
                    b = self.adj_matrix[p].nonzero()
                    #b[0]存储的是邻接矩阵中p节点所在行的非零元素所在的行
                    #这里只是为了计算p节点所在行的非零元素个数
                    b = b[0]
#先从原图中随机抽取节点作为起点，然后从该节点的邻居节点中随机抽取一个节点来构成子图
                    w = int(random.random() * len(b))
                    new_adj_matrix[p][b[w]] = 1
                    new_adj_matrix[b[w]][p] = 1
                    new_links.append([p,b[w],1])
                    #将抽取到的邻居节点作为下一次循环的起点
                    p = b[w]
            #self.order就存放了所有随机游走所经历过的点，并且不会重复存储
            self.order = self.order.keys()
            self.adj_matrix = new_adj_matrix
            self.N = new_N
            self.links = new_links
            print (len(self.links))
            return self
            pass
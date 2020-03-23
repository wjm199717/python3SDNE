'''
Reference implementation of SDNE
Author: Xuanrong Yao, Daixin wang
for more detail, refer to the paper:
SDNE : structral deep network embedding
Wang, Daixin and Cui, Peng and Zhu, Wenwu
Knowledge Discovery and Data Mining (KDD), 2016
'''

#!/usr/bin/python2
# -*- coding: utf-8 -*-


import numpy as np
from config import Config
from graph import Graph
from sdne import SDNE
import util
#从scipy包/库 中导入模块io
#scipy是包/库 的目录名，或者叫高级模块，这个高级模块里面包含了很多模块
#scipy是所有模块组成的包的名字，仅仅是名字而已
#io是包/库 中的模块名，该模块用来控制数据输入输出
import scipy.io as sio
#time模块是用来控制时间的
import time
#optparse用于处理命令行参数
#optparse模块主要用来为脚本传递命令参数，
#采用预先定义好的选项来解析命令行参数。
from optparse import OptionParser
#os模块包含普遍的操作系统功能
import os

if __name__ == "__main__":
    #下面6句代码是来指定所用的数据集文件
    #设置当前使用的GPU设备仅为1号设备  设备名称为'/gpu:1'
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    parser = OptionParser()
    #add_option用来加入选项，dest是存储的变量形参关键字
    #'-c'是命令行参数值，"config_file"是变量名
    parser.add_option("-c",dest = "config_file", action = "store", metavar = "CONFIG FILE")
    #指定变量的值，即'config_file'的值为'blogcatalog.ini'
    args = ['-c', 'blogcatalog.ini']
    #optinos是存储变量也就是dest的对象
    #调用options.config_file就可以得到一个值
    #parser.parse_args()解析的作用
    options, _ = parser.parse_args(args)
    if options.config_file == None:
        raise IOError("no config file specified")
    
    #config实例化对象中存储的是3个数据集的各种信息
    #获取已经指定的数据集文件中的内容
    config = Config(options.config_file)
    
    #获得图数据
    train_graph_data = Graph(config.train_graph_file, config.ng_sample_ratio)
    
    #训练图跟原始图不一定相等
    if config.origin_graph_file:
        origin_graph_data = Graph(config.origin_graph_file, config.ng_sample_ratio)

    if config.label_file:
        #load label for classification
        #会为所有节点得到一个标签矩阵
        train_graph_data.load_label_data(config.label_file)
    
    #改变构造的神经网络结构输入层也就是第一层
    config.struct[0] = train_graph_data.N
    
    model = SDNE(config)
    #对神经网络要学习的参数进行初始化，这里不是随机初始化而是rbm初始化
    #因为要用到rbm,所以用rbm初始化神经网络之前，需要对rbm进行训练，
    #再用训练好的rbm结构来对神经网络进行初始化
    model.do_variables_init(train_graph_data)
    embedding = None
    #下面进行0次迭代结果，也就是rbm初始化神经网络产生的embedding
    while (True):
#按照顺序对数据进行采样，每次采一个批次，依次进行，有顺序的，一直到所有数据集都被采完
#进行目标函数优化时就是这每一个批次对参数优化一次
        mini_batch = train_graph_data.sample(config.batch_size, do_shuffle = False)
        if embedding is None:
            #这里的mini_batch是数据集的一个批次
            embedding = model.get_embedding(mini_batch)
        else:
            #np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组
            #将前一个批次的embedding和后一个批次的embedding按行堆叠在一起
            #[[[],[],[]...,[]],[[],[],...,[]]],张量起步会有一层[]，表示是张量
#此时的embedding是通过rbm初始化神经网络所得到的embedding，并不是训练好的神经网络所得出的embedding
            embedding = np.vstack((embedding, model.get_embedding(mini_batch))) 
        if train_graph_data.is_epoch_end:
            break
    #此时train_graph_data.is_epoch_end为true
    #此时epoch0迭代完毕

    epochs = 0
    batch_n = 0
    
    #tt存储的是当前运行该模块的具体时间
#    tt = time.ctime().replace(' ','-')
    path = "./result/" + config.embedding_filename
#system函数可以将字符串转化成命令在服务器上运行
    #os.mkdir创建目录
    os.mkdir(path)
    #fout是以写的形式打开的文件对象 
    fout = open(path + "/log.txt","w")
    model.save_model(path + '/epoch0.model')
    #在python中可以使用scipy.io中的函数loadmat()读取mat文件，函数savemat保存文件。
    #mat文件中数据读出来和保存的形式都是字典形式
        #这里的embedding应该是一个3维的张量
        #这里的embedding不是最终的embedding,这里是rbm初始化神经网络得到的embedding
    sio.savemat(path + '/embedding.mat',{'embedding':embedding})
    print ("!!!!!!!!!!!!!")
    while (True):
        if train_graph_data.is_epoch_end:
            loss = 0
            if epochs % config.display == 0:
                embedding = None
                while (True):
                    #执行一次sample就采一个批次的数据，有顺序的执行
                    mini_batch = train_graph_data.sample(config.batch_size, do_shuffle = False)
    #下面这句话有问题还值得思考
    #第一个批次的损失会加到第二个批次的损失上，所以损失是累加的，累加是因为要记录一次迭代所产生的全部损失
                    loss += model.get_loss(mini_batch)
                    if embedding is None:
                        embedding = model.get_embedding(mini_batch)
                    else:
                        embedding = np.vstack((embedding, model.get_embedding(mini_batch))) 
                    if train_graph_data.is_epoch_end:
                        break
    
                print ("Epoch : %d loss : %.3f" % (epochs, loss))
                fout.write(" Epoch : %d loss : %.3f " % (epochs, loss))
                #以下指标都是用来衡量embedding的好坏，指标数值越高模embedding越好
                #embedding越好说明模型就越好
                if config.check_reconstruction:
                    fout.write(str(epochs) + "reconstruction:" + str(util.check_reconstruction(embedding, train_graph_data, config.check_reconstruction)))
                if config.check_link_prediction:
                    fout.write(str(epochs) + " link_prediction:" + str(util.check_link_prediction(embedding, train_graph_data, origin_graph_data, config.check_link_prediction)))
                if config.check_classification:
                    data = train_graph_data.sample(train_graph_data.N, do_shuffle = False,  with_label = True)
                    fout.write(str(epochs) + " classification" + str(util.check_multi_label_classification(embedding, data.label)))
     #flush() 方法是用来刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区，
                fout.flush()
                model.save_model(path + '/epoch' + str(epochs) + ".model")
            if epochs == config.epochs_limit:
                print ("exceed epochs limit terminating")
                break
            epochs += 1
            #一个数据集一次迭代没有全部训练完，就先采样得到一个批次的数据，然后记录损失，然后优化损失
        mini_batch = train_graph_data.sample(config.batch_size)
    
    #model.fit就记录采样采到某一批次的损失并优化该损失更新参数，一直到一个数据集一次迭代完毕
    #一次迭代又分很多批次进行
    #每次优化一个批次的损失，优化的损失不会随着批次数增加而累加之前的批次损失
        loss = model.fit(mini_batch)
    
    sio.savemat(path + '/embedding.mat',{'embedding':embedding})
    fout.close()
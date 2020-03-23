import tensorflow as tf
import numpy as np

class rbm:
    def __init__(self, shape, para):
#特征的输入层是可视神经层，一个rbm的隐藏层会成为下一个rbm的可视层也就是输入层
        # shape[0] means the number of visible units
        # shape[1] means the number of hidden units
        self.para = para
        self.sess = tf.Session()
        #stddev是标准差
        #np.sqrt计算数组各元素的平方根
        #shape[0]这里的含义就是通指每个rbm的可视层
        stddev = 1.0 / np.sqrt(shape[0])
        self.W = tf.Variable(tf.random_normal([shape[0], shape[1]], stddev = stddev), name = "Wii")
        #rbm的可视层跟隐藏层都会有偏差，因为下一个rbm的可视层是上一个rbm的隐藏层
        self.bv = tf.Variable(tf.zeros(shape[0]), name = "a")
        self.bh = tf.Variable(tf.zeros(shape[1]), name = "b")
        #None表示样本数是不确定的，可以为任意值
        self.v = tf.placeholder("float", [None, shape[0]])
        #这里只是定义了变量初始化，还没激活，激活需要sess.run来激活
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.buildModel()
        print ("rbm init completely")

    def buildModel(self):
        #self.h是一个2维张量
        #tf.matmul是矩阵的乘法，这不是矩阵的点乘
        self.h = self.sample(tf.sigmoid(tf.matmul(self.v, self.W) + self.bh))
        #gibbs_sample
        v_sample = self.sample(tf.sigmoid(tf.matmul(self.h, tf.transpose(self.W)) + self.bv))
        h_sample = self.sample(tf.sigmoid(tf.matmul(v_sample, self.W) + self.bh))
        #tf.to_float()将张量强制转换为float32类型。
        lr = self.para["learning_rate"] / tf.to_float(self.para["batch_size"])
        #assign_add(a,b)通过增加b的值来更新a的值
        W_adder = self.W.assign_add(lr  * (tf.matmul(tf.transpose(self.v), self.h) - tf.matmul(tf.transpose(v_sample), h_sample)))
        bv_adder = self.bv.assign_add(lr * tf.reduce_mean(self.v - v_sample, 0))
        bh_adder = self.bh.assign_add(lr * tf.reduce_mean(self.h - h_sample, 0))
        self.upt = [W_adder, bv_adder, bh_adder]
        self.error = tf.reduce_sum(tf.pow(self.v - v_sample, 2))
    
    def fit(self, data):
        #ret存储的是self.error
        _, ret = self.sess.run((self.upt, self.error), feed_dict = {self.v : data})
        return ret
    
    def sample(self, probs):
        #tf.floor()是参数里的张量的元素全部向下取整数，比如3.6就等于3
        #tf.random_uniform是均匀分布随机产生值，值位于0-1之间，
        return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))
    def getWb(self):
        #返回的是一个列表，列表中包括有三个元素
        return self.sess.run([self.W, self.bv, self.bh])
    def getH(self, data):
        #返回的是self.h的值
        return self.sess.run(self.h, feed_dict = {self.v : data})

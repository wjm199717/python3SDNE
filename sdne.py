import numpy as np
import tensorflow as tf
import rbm 

class SDNE:
    def __init__(self, config):
    
        self.is_variables_init = False
        self.config = config 
        ######### not running out gpu sources ##########
#tf.ConfigProto()主要的作用是配置tf.Session的运算方式，比如gpu运算或者cpu运算
        tf_config = tf.ConfigProto()
#当使用GPU时候，Tensorflow运行自动慢慢达到最大GPU的内存
        tf_config.gpu_options.allow_growth = True
#设置session的运行使用GPU来运行
        self.sess = tf.Session(config =  tf_config)

        ############ define variables ##################
        #构建几层的神经网络
        self.layers = len(config.struct)
        self.struct = config.struct
        #config.sparse_dot = False
        self.sparse_dot = config.sparse_dot
        self.W = {}
        self.b = {}
        #struct存储的是网络中每层神经元的个数
        struct = self.struct
#构建的神经网络层数把输入层算进去了，所以真正有权重偏置的只有两层
        for i in range(self.layers - 1):
            name = "encoder" + str(i)
            #输入层是一个节点邻接矩阵所属的那一行，所以也有全部节点个数的元素
            self.W[name] = tf.Variable(tf.random_normal([struct[i], struct[i+1]]), name = name)
            self.b[name] = tf.Variable(tf.zeros([struct[i+1]]), name = name)
        struct.reverse()
        for i in range(self.layers - 1):
            name = "decoder" + str(i)
            self.W[name] = tf.Variable(tf.random_normal([struct[i], struct[i+1]]), name = name)
            self.b[name] = tf.Variable(tf.zeros([struct[i+1]]), name = name)
        self.struct.reverse()
        ###############################################
        ############## define input ###################
        
        #邻接矩阵的占位符，邻接矩阵是二维的
        self.adjacent_matriX = tf.placeholder("float", [None, None])
        # these variables are for sparse_dot
        self.X_sp_indices = tf.placeholder(tf.int64)
        self.X_sp_ids_val = tf.placeholder(tf.float32)
        self.X_sp_shape = tf.placeholder(tf.int64)
        #稀疏张量，第一个参数表示非零元素的位置，第二个参数表示非零元素的值，
        #第三个参数表示系数张量的shape
        self.X_sp = tf.SparseTensor(self.X_sp_indices, self.X_sp_ids_val, self.X_sp_shape)
        #输入占位符的形式是[[a节点的邻接行],[b节点的邻接行],[c节点的邻接行]]
        #这里的None是说数字没有限制
        self.X = tf.placeholder("float", [None, config.struct[0]])
        
        ###############################################
        #__开头表明这个函数是私有的
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.__make_compute_graph()
        self.loss = self.__make_loss(config)
        self.optimizer = tf.train.RMSPropOptimizer(config.learning_rate).minimize(self.loss)
        

    
    #一个神经网络就可以看作是一个图，所以有时候用‘图’这个名字来表示神经网络
    #猜想可能是因为神经网络的结构在tensorboard中是以图的形式展示的，所以取名为‘图’
    def __make_compute_graph(self):
        def encoder(X):
            for i in range(self.layers - 1):
                name = "encoder" + str(i)
                X = tf.nn.sigmoid(tf.matmul(X, self.W[name]) + self.b[name])
            return X

        def encoder_sp(X):
            for i in range(self.layers - 1):
                name = "encoder" + str(i)
                if i == 0:
                #猜想如果输入是一个稀疏矩阵，那么采用这种方法可能提高计算效率
                    X = tf.nn.sigmoid(tf.sparse_tensor_dense_matmul(X, self.W[name]) + self.b[name])
                else:
                    X = tf.nn.sigmoid(tf.matmul(X, self.W[name]) + self.b[name])
            return X
            
        def decoder(X):
            for i in range(self.layers - 1):
                name = "decoder" + str(i)
                X = tf.nn.sigmoid(tf.matmul(X, self.W[name]) + self.b[name])
            return X
            
        if self.sparse_dot:
            self.H = encoder_sp(self.X_sp)
        else:
            self.H = encoder(self.X)
        self.X_reconstruct = decoder(self.H)
    

        
    def __make_loss(self, config):
        def get_1st_loss_link_sample(self, Y1, Y2):
            #pow返回（Y1-Y2）的平方，最后将每一个元素都累加起来
            return tf.reduce_sum(tf.pow(Y1 - Y2, 2))
        def get_1st_loss(H, adj_mini_batch):
            #tf.reduce_sum(_)直接得到一个数
            #tf.reduce_sum(_,1)是按照行的方向来累加,每行返回一个数，张量维度减少1
            #tf.reduce_sum(_,0)是按照列的方向来累加,每列返回一个数，张量维度减少1
            #tf.diag(_)是用给定的值来返回一个对角矩阵
            #D是一个度矩阵，也是一个对角矩阵
            D = tf.diag(tf.reduce_sum(adj_mini_batch,1))
            #度矩阵减去邻接矩阵等于拉普拉斯矩阵
            L = D - adj_mini_batch ## L is laplation-matriX
            #trace(x) 返回沿 x 中每个最内层矩阵的主对角线的总和.
            #一阶相似性优化问题可以转换为下面这种矩阵向量表示形式
            #拉普拉斯特征映射算法：两个实例很相似那么在降维空间里这两个实例也很接近
            return 2*tf.trace(tf.matmul(tf.matmul(tf.transpose(H),L),H))

        def get_2nd_loss(X, newX, beta):
            B = X * (beta - 1) + 1
            return tf.reduce_sum(tf.pow((newX - X)* B, 2))

        def get_reg_loss(weight, biases):
            #正则化项，保证所有参数尽可能小
            #itervalues()会在迭代过程中依次从 dict 中取出 value，相比values()而言，更节约内存
            #values()会生成一个列表，而itervalues()不会
#tf.add_n就是将一个张量里面的元素相加，实现一个列表的元素的相加，2维变1维，1维变一个数
            ret = tf.add_n([tf.nn.l2_loss(w) for w in weight.values()])
            ret = ret + tf.add_n([tf.nn.l2_loss(b) for b in biases.values()])
            return ret
            
        #Loss function
        self.loss_2nd = get_2nd_loss(self.X, self.X_reconstruct, config.beta)
        self.loss_1st = get_1st_loss(self.H, self.adjacent_matriX)
        self.loss_xxx = tf.reduce_sum(tf.pow(self.X_reconstruct,2)) 
        # we don't need the regularizer term, since we have nagetive sampling.
        self.loss_reg = get_reg_loss(self.W, self.b) 
        return config.gamma * self.loss_1st + config.alpha * self.loss_2nd + config.reg * self.loss_reg
        
        #return config.gamma * self.loss_1st + config.alpha * self.loss_2nd +self.loss_xxx

    def save_model(self, path):
        #self.b和self.W都是字典
        #tf.train.Saver保存指定的变量，参数可以是一个字典也可以是一个列表（指定要保存的东西）
        saver = tf.train.Saver(list(self.b.values()) + list(self.W.values()))
        saver.save(self.sess, path)

    def restore_model(self, path):
        saver = tf.train.Saver(list(self.b.values()) + list(self.W.values()))
        saver.restore(self.sess, path)
        self.is_Init = True
    
    def do_variables_init(self, data):
        def assign(a, b):
            #将b的值赋给a
            op = a.assign(b)
            self.sess.run(op)
        init = tf.global_variables_initializer()       
        self.sess.run(init)
        #如果配置文件的模型加载地址存在，就加载已经保存好了的模型
        if self.config.restore_model:
            self.restore_model(self.config.restore_model)
            print ("restore model" + self.config.restore_model)
        elif self.config.DBN_init:
            #self.struct存储的是神经网络的结构，每一层多少个神经元
            #例如3层神经网络就是2个rbm
            shape = self.struct
            myRBMs = []
            #i代表第几个rbm
            for i in range(len(shape) - 1):
                myRBM = rbm.rbm([shape[i], shape[i+1]], {"batch_size": self.config.dbn_batch_size, "learning_rate":self.config.dbn_learning_rate})
                myRBMs.append(myRBM)
                for epoch in range(self.config.dbn_epochs):
                    error = 0
                    #训练一个批次的样本就更新一次参数
                    for batch in range(0, data.N, self.config.dbn_batch_size):
                        mini_batch = data.sample(self.config.dbn_batch_size).X
                        for k in range(len(myRBMs) - 1):
                            mini_batch = myRBMs[k].getH(mini_batch)
                        #一个批次计算一次错误也就是损失，然后更新一次参数
                        error += myRBM.fit(mini_batch)
                    #记录错误是一次迭代记录一次错误
                    print ("rbm epochs:", epoch, "error : ", error)

                W, bv, bh = myRBM.getWb()
                name = "encoder" + str(i)
                assign(self.W[name], W)
                assign(self.b[name], bh)
                name = "decoder" + str(self.layers - i - 2)
                assign(self.W[name], W.transpose())
                assign(self.b[name], bv)
        self.is_Init = True

    def __get_feed_dict(self, data):
        #得到喂给图的真实数据
        #以真实数据来替代之前定义好的占位符
        X = data.X
        if self.sparse_dot:
            #np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组
            #astype()实现数组中的元素类型转换
            #.T对一个矩阵的转置
            #np.where()返回满足条件的下标列表
            X_ind = np.vstack(np.where(X)).astype(np.int64).T
            X_shape = np.array(X.shape).astype(np.int64)
            X_val = X[np.where(X)]
            #往占位符里面输入真实数据
            return {self.X : data.X, self.X_sp_indices: X_ind, self.X_sp_shape:X_shape, self.X_sp_ids_val: X_val, self.adjacent_matriX : data.adjacent_matriX}
        else:
            return {self.X: data.X, self.adjacent_matriX: data.adjacent_matriX}
            
    def fit(self, data):
        #实施喂养/替代操作
        #开始实施将真实数据喂给图的这个操作
        feed_dict = self.__get_feed_dict(data)
        #ret存储的是self.loss的值，_存储的是self.optimizer的值
        #得到该损失函数，然后对该损失函数进行优化
        ret, _ = self.sess.run((self.loss, self.optimizer), feed_dict = feed_dict)
        return ret
    
    def get_loss(self, data):
        feed_dict = self.__get_feed_dict(data)
        #返回的是self.loss的值
        #sess.run()返回的是它的第一个参数的值
        return self.sess.run(self.loss, feed_dict = feed_dict)

    def get_embedding(self, data):
        #self.sess.run()返回的就是self.H的值
        #得到嵌入向量
        return self.sess.run(self.H, feed_dict = self.__get_feed_dict(data))

    def get_W(self):
        #得到权重参数的值
        return self.sess.run(self.W)
        
    def get_B(self):
        #得到偏置参数的值
        return self.sess.run(self.b)
        
    def close(self):
        #如果会话是以with的形式打开则不需要关闭会话，系统会自动关闭会话
        #否则需要自己手动去关闭会话，所以说会话也是需要关闭的
        self.sess.close()
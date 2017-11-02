#!/usr/bin/env python
# coding=utf-8
#by fukaiyu
import tensorflow as tf
import math
from time import clock
import numpy as np
import sys
import os
import pickle
import sys
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from datetime import datetime
from util import *
from datetime import datetime

class DeepFM():
    def __init__(self, field, feature, param=None):
        '''
            模型类
            输入：
            field，field个数，类型int
            feature，feature个数，类型int
            param，模型相关配置，类型dict{
                reg_w_linear: 一阶项正则，类型float
                reg_w_fm: 二阶项正则，类型float
                reg_w_nn: DNN权重正则，类型float
                reg_w_l1: L1正则，类型float
                init_value：初始化权重的标准差，类型float
                layer_sizes：DNN每一层的神经元个数，类型list[int, int, ...] 对应层数
                acitivations：DNN每一层的激活函数，类型list[string, string, ...] 对应层数
                eta：学习率，类型float 优化器为adadelta的时学习率
                n_epoch：训练周期数，类型int
                batch_size：batch大小，类型int
                dim：FM分解向量的维度，类型int
                output_predictions
                is_use_fm_part：是否使用fm模型，类型bool
                is_use_dnn_part：是否使用dnn模型，类型bool
                learning_rate：学习率，类型float 优化器为adadelta以外的学习率
                loss：损失函数，类型: string, ie:[cross_entropy_loss, square_loss, log_loss]
                optimizer：优化方式，类型：string ie:[adam, ftrl, sgd]
                log_dir: 训练信息和日志存储路径，类型string
            }
        '''
        self.numOfField = field
        self.numOfFeature  = feature
        if param == None:
            self.param = {
                'reg_w_linear': 0.00010, 'reg_w_fm':0.0001, 'reg_w_nn': 0.0001,  #0.001
                'reg_w_l1': 0.0001,
                'init_value': 0.1,
                'layer_sizes': [30, 10],
                'activations':['tanh','tanh'],
                'eta': 0.1,
                'n_epoch': 5000,  # 500
                'batch_size': 50,
                'dim': 15,
                'output_predictions':False,
                'is_use_fm_part':True,
                'is_use_dnn_part':True,
                'learning_rate':0.01, # [0.001, 0.01]
                'loss': 'log_loss', # [cross_entropy_loss, square_loss, log_loss]
                'optimizer':'sgd', # [adam, ftrl, sgd]
                'log_dir': './logs/' + datetime.utcnow().strftime('%Y-%m-%d_%H_%M_%S') #tensorboard日志位置
            }
        else:
            self.param = param
        self.optimizer = self.param['optimizer']
        self.epoch = self.param['n_epoch']
        self.batch_size = self.param['batch_size']
        self.init_value = self.param['init_value']
        self.layer_sizes = self.param['layer_sizes']
        self.dim = self.param['dim']
        self.is_use_fm_part =  self.param['is_use_fm_part']
        self.is_use_dnn_part = self.param['is_use_dnn_part']
        self.activations = self.param['activations']
        self.lossf = self.param['loss']
        self.reg_w_linear = self.param['reg_w_linear']
        self.reg_w_fm = self.param['reg_w_fm']
        self.reg_w_nn = self.param['reg_w_nn']
        self.reg_w_l1 = self.param['reg_w_l1']
        self.log_dir = self.param['log_dir']
        self.learning_rate = self.param['learning_rate']
        tf.reset_default_graph()
        self._indices = tf.placeholder(tf.int64, shape=[None, 2], name='raw_indices')
        self._values = tf.placeholder(tf.float32, shape=[None], name='raw_values')
        self._values2 = tf.placeholder(tf.float32, shape=[None], name='raw_values_square')
        self._shape = tf.placeholder(tf.int64, shape=[2], name='raw_shape')

        self._y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')
        self._ind = tf.placeholder(tf.int64, shape=[None])


    def BuildModel(self):
        '''
            构建模型，最后返回一个更新参数的操作句柄、错误率、损失、每个实例的分数、tensorboard画图句柄
        '''
        eta = tf.constant(self.param['eta'])
        _x = tf.SparseTensor(self._indices, self._values, self._shape)
        _xx = tf.SparseTensor(self._indices, self._values2, self._shape)

        model_params = [] # deep + fm 的所有参数 w & b
        self.tmp = []

        fm_w_linear = tf.Variable(tf.truncated_normal([self.numOfFeature, 1], stddev=self.init_value, mean=0), name='fm_w_linear', dtype=tf.float32) # 初始化fm部分的一阶线性权重w，截尾正态分布，每个feature一个权重
        fm_bias = tf.Variable(tf.truncated_normal([1], stddev=self.init_value, mean=0), name='fm_bias') # 初始化fm部分的偏置即w0

        model_params.append(fm_bias)
        model_params.append(fm_w_linear)

        self.fm_output = fm_bias + tf.sparse_tensor_dense_matmul(_x, fm_w_linear, name='fm_output')

        fm_w_vector = tf.Variable(tf.truncated_normal([self.numOfFeature, self.dim], stddev=self.init_value / math.sqrt(float(self.dim)), mean=0), name='fm_w_vector', dtype=tf.float32) # 初始化fm部分的二阶线性权重w，截尾正态分布，每个feature用一个dim维的vector表示，且每个vector的L1范数初始化标准差为init_value
        model_params.append(fm_w_vector)

        if self.is_use_fm_part: # 如果要使用二阶项
            self.fm_output += 0.5 * tf.reduce_sum(tf.pow(tf.sparse_tensor_dense_matmul(_x, fm_w_vector), 2) - tf.sparse_tensor_dense_matmul(_xx, tf.pow(fm_w_vector, 2)), 1, keep_dims=True)

        if self.is_use_dnn_part: # 如果要使用DNN
            w_fm_nn_input = tf.reshape(tf.gather(fm_w_vector, self._ind) * tf.expand_dims(self._values, 1), [-1, self.numOfField * self.dim]) # feature的v向量作为作为权重，在每一个field中只有一个x会为1，因此只有与那个x连接的权重才会对embedding层的output产生影响,embedding层的输出，大小为:[batch_size, filed_size * self.dim]
            print(w_fm_nn_input.shape)
            self.tmp.append(tf.shape(tf.expand_dims(self._values, 1)))
            self.tmp.append(tf.shape(w_fm_nn_input))
            self.tmp.append(tf.shape(tf.gather(fm_w_vector, self._ind) * tf.expand_dims(self._values, 1)))
            self.tmp.append(tf.shape(tf.gather(fm_w_vector, self._ind)))

            hidden_nn_layers = []
            hidden_nn_layers.append(w_fm_nn_input)
            last_layer_size = self.numOfField * self.dim
            layer_idx = 0
            w_nn_params = [] # DNN中的参数w
            b_nn_params = [] # DNN中的参数b

            for layer_size in self.layer_sizes: # 全连接层
                cur_w_nn_layer = tf.Variable(tf.truncated_normal([last_layer_size, layer_size], stddev=self.init_value / math.sqrt(float(layer_size)), mean=0), name='w_nn_layer' + str(layer_idx), dtype=tf.float32)
                cur_b_nn_layer = tf.Variable(tf.truncated_normal([layer_size], stddev=self.init_value, mean=0), name='b_nn_layer' + str(layer_idx))
                cur_hidden_nn_layer = tf.nn.xw_plus_b(hidden_nn_layers[layer_idx], cur_w_nn_layer, cur_b_nn_layer)

                if self.activations[layer_idx] == 'tanh':
                    cur_hidden_nn_layer = tf.nn.tanh(cur_hidden_nn_layer)
                elif self.activations[layer_idx] == 'sigmoid':
                    cur_hidden_nn_layer = tf.nn.sigmoid(cur_hidden_nn_layer)
                elif self.activations[layer_idx] == 'relu':
                    cur_hidden_nn_layer = tf.nn.relu(cur_hidden_nn_layer)

                hidden_nn_layers.append(cur_hidden_nn_layer)
                layer_idx += 1
                last_layer_size = layer_size

                model_params.append(cur_w_nn_layer)
                model_params.append(cur_b_nn_layer)

                w_nn_params.append(cur_w_nn_layer)
                b_nn_params.append(cur_b_nn_layer)

            w_nn_output = tf.Variable(tf.truncated_normal([last_layer_size, 1], stddev=self.init_value, mean=0), name='w_nn_output', dtype=tf.float32) # DNN输出层的权重
            nn_output = tf.matmul(hidden_nn_layers[-1], w_nn_output) # DNN最后一层的输出

            model_params.append(w_nn_output)
            w_nn_params.append(w_nn_output)

            self.fm_output += nn_output

        if self.lossf == 'cross_entropy_loss':
            self.error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(self.fm_output, [-1]), labels=tf.reshape(_y, [-1])))

        elif self.lossf == 'square_loss':
            self.error = tf.reduce_mean(tf.squared_difference(tf.sigmoid(self.fm_output), self._y))

        elif self.lossf == 'log_loss':
            self.error = tf.reduce_mean(tf.losses.log_loss(predictions=tf.sigmoid(self.fm_output), labels=self._y))

        lambda_w_linear = tf.constant(self.reg_w_linear, name='lambda_w_linear')
        lambda_w_fm = tf.constant(self.reg_w_fm, name='lambda_w_fm')
        lambda_w_nn = tf.constant(self.reg_w_nn, name='lambda_w_nn')
        lambda_w_l1 = tf.constant(self.reg_w_l1, name='lambda_w_l1')

        l2_norm = tf.multiply(lambda_w_linear, tf.reduce_sum(tf.pow(fm_w_linear, 2))) # 一阶项正则
        l2_norm = tf.multiply(lambda_w_l1, tf.reduce_sum(tf.abs(fm_w_linear))) # 一阶项L1正则

        if self.is_use_fm_part  or self.is_use_dnn_part: # fm二阶项正则
            l2_norm += tf.multiply(lambda_w_fm, tf.reduce_sum(tf.pow(fm_w_vector, 2)))

        if self.is_use_dnn_part: # dnn权重和偏置正则
            for i in range(len(w_nn_params)):
                l2_norm += tf.multiply(lambda_w_nn, tf.reduce_sum(tf.pow(w_nn_params[i], 2)))

            for i in range(len(b_nn_params)):
                l2_norm += tf.multiply(lambda_w_nn, tf.reduce_sum(tf.pow(b_nn_params[i], 2)))

        self.loss = tf.add(self.error, l2_norm)
        if self.optimizer == 'adadelta':
            self.train_step = tf.train.AdadeltaOptimizer(eta).minimize(self.loss, var_list=model_params)
        elif self.optimizer == 'sgd':
            self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, var_list=model_params)
        elif self.optimizer == 'adam':
            self.train.step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=model_params)
        elif self.optimizer =='ftrl':
            self.train.step = tf.train.FtrlOptimizer(self.learning_rate).minimize(self.loss, var_list=model_params)
        else:
            self.train.step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, var_list=model_params)

        tf.summary.scalar('square_error', self.error)
        tf.summary.scalar('loss', self.loss)
        tf.summary.histogram('linear_weights_hist', fm_w_linear)

        if self.is_use_fm_part:
            tf.summary.histogram('fm_weights_hist', fm_w_vector)

        if self.is_use_dnn_part:
            for idx in range(len(w_nn_params)):
                tf.summary.histogram('nn_layer' + str(idx), w_nn_params[idx])

        self.merged_summary = tf.summary.merge_all()
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)


        return self.train_step, self.loss, self.error, self.fm_output, self.merged_summary, self.tmp

    def Fit(self, train, test, full=1, instances=0):
        '''
            训练函数
            输入：train，训练集路径，类型String
            test，测试集路径，类型String
            full，是否读取整个数据集
            instances，当full为0时，读取的数据集实例数
        '''
        if full == 1: #读取整个数据集
            pre_build_data_cache_if_need(train, self.numOfFeature, self.batch_size)
            pre_build_data_cache_if_need(test, self.numOfFeature, self.batch_size)
        else : #读取部分数据集并分割
            pre_build_some_data_cache_if_need(train, test, instances, self.numOfFeature, self.batch_size)
            train += '.txt'
            test += '.txt'


        print 'beginning running'

        saver = tf.train.Saver()

        log_writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)
        glo_ite = 0 # 总迭代次数

        for epoch in range(self.epoch):
            iteration = -1
            start = clock()

            time_load_data, time_sess = 0, 0
            time_cp02 = clock()

            train_loss_per_epoch = 0

            for training_input_in_sp in load_data_cache(train.replace('.csv', '.pkl').replace('.txt', '.pkl')):
                time_cp01 = clock()
                time_load_data += time_cp01 - time_cp02
                iteration += 1

                # print self.sess.run(self.tmp, feed_dict={self._indices: training_input_in_sp['indices'], self._values: training_input_in_sp['values'], self._shape: training_input_in_sp['shape'], self._y: training_input_in_sp['labels'], self._values2: training_input_in_sp['values2'], self._ind: training_input_in_sp['feature_indices']})
                _, cur_loss, error, score, summary = self.sess.run([self.train_step, self.loss, self.error, self.fm_output, self.merged_summary], feed_dict={self._indices: training_input_in_sp['indices'], self._values: training_input_in_sp['values'], self._shape: training_input_in_sp['shape'], self._y: training_input_in_sp['labels'], self._values2: training_input_in_sp['values2'], self._ind: training_input_in_sp['feature_indices']})
                time_cp02 = clock()
                time_sess += time_cp02 - time_cp01

                train_loss_per_epoch += cur_loss

                log_writer.add_summary(summary, glo_ite)
            end = clock()

            if epoch % 5 == 0:
                auc = self.Predict(test.replace('.csv', '.pkl').replace('.txt', '.pkl'))
                print ('auc is', auc, ', at epoch ', epoch, ', time is {0:.4f} min'.format((end - start) / 60.0), ', train_loss is {0:2f}'.format(train_loss_per_epoch))

        log_writer.close()


xxx_test    def Predict(self, test_file):
        gt_scores = []
        pred_scores = []
        for test_input_in_sp in load_data_cache(test_file):
            score = self.sess.run(self.fm_output, feed_dict={self._indices: test_input_in_sp['indices'], self._values: test_input_in_sp['values'], self._shape: test_input_in_sp['shape'], self._y: test_input_in_sp['labels'],
                                                                   self._values2: test_input_in_sp['values2'], self._ind: test_input_in_sp['feature_indices']})
            gt_scores.extend(test_input_in_sp['labels'].reshape(-1).tolist())
            pred_scores.extend(score)


        auc = roc_auc_score(np.asarray(gt_scores), np.asarray(pred_scores))

        return auc

if __name__ == '__main__':
    clf = DeepFM(140, 140)
    clf.BuildModel()
    clf.Fit('xxx.txt', 'xxx_test.txt' )

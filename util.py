#!/usr/bin/env python
# coding=utf-8
import os
import pickle
import numpy as np

def pre_build_data_cache_if_need(infile, feature_cnt, batch_size):
    '''
        数据集序列化
    '''
    outfile = infile.replace('.csv','.pkl').replace('.txt','.pkl')
    if not os.path.isfile(outfile):
        print('pre_build_data_cache for ', infile)
        pre_build_data_cache(infile, outfile, feature_cnt, batch_size)
        print('pre_build_data_cache finished.' )

def pre_build_some_data_cache_if_need(train, test, instances, feature_cnt, batch_size):
    '''
        数据集序列化
    '''
    outfile = train + '.pkl'
    test_outfile = test + '.pkl'
    i = 0
    if not os.path.isfile(outfile):
        with open(outfile, 'wb') as wt, open(test_outfile, 'wb') as we:
            for labels, features in load_data_from_file_batching_2(train, batch_size, instances):
                input_in_sp = prepare_data_4_sp(labels, features, feature_cnt)
                if i < 0.8 * instances / batch_size:
                    pickle.dump(input_in_sp, wt)
                else: 
                    pickle.dump(input_in_sp, we)
                i += 1



        
def load_data_from_file_batching_2(file, batch_size, instances):
    labels = []
    features = []
    cnt = 0
    total = 0
    with open(file, 'r') as rd:
        while total <= instances:
            line = rd.readline()
            if not line:
                break
            cnt += 1
            if '#' in line: # 文件读取到#符号为止
                punc_idx = line.index('#')
            else:
                punc_idx = len(line)
            label = float(line[0:1])
            if label>1:
                label=1
            feature_line = line[2:punc_idx]
            words = feature_line.split(' ')
            if len(words) == 1:
                words = feature_line.split(',')
            cur_feature_list = []
            try:
                for word in words:
                    if not word:
                        continue
                    tokens = word.split(':')
                    if len(tokens[1]) <= 0:
                        tokens[1] = '0'
                    cur_feature_list.append([int(tokens[0]), float(tokens[1])])
                features.append(cur_feature_list)
                labels.append(label)
            except:
                raise Exception
                total -= 1
                cnt -= 1
            total += 1
            if cnt == batch_size:
                yield labels, features
                labels = []
                features = []
                cnt = 0


def pre_build_data_cache(infile, outfile, feature_cnt, batch_size):
    '''
        预缓存数据，feature类型: [feature序号，feature值] [label值]
    '''
    wt = open(outfile, 'wb')
    for labels, features in load_data_from_file_batching(infile, batch_size):
        input_in_sp = prepare_data_4_sp(labels, features, feature_cnt)
        pickle.dump(input_in_sp, wt)
    wt.close()

def load_data_from_file_batching(file, batch_size):
    '''
        按批从文件读取数据
        数据存储格式为：label feature编号:feature值 ...... #注释
    '''
    labels = []
    features = []
    cnt = 0
    with open(file, 'r') as rd:
        while True:
            line = rd.readline()
            if not line:
                break
            cnt += 1
            if '#' in line: # 文件读取到#符号为止
                punc_idx = line.index('#')
            else:
                punc_idx = len(line)
            label = float(line[0:1])
            if label>1:
                label=1
            feature_line = line[2:punc_idx]
            words = feature_line.split(' ')
            if len(words) == 1:
                words = feature_line.split(',')
            cur_feature_list = []
            for word in words:
                if not word:
                    continue
                tokens = word.split(':')

                # if tokens[0]=='4532':
                #    print('line ', cnt, ':    ',word, '    line:', line)
                if len(tokens[1]) <= 0:
                    tokens[1] = '0'
                cur_feature_list.append([int(tokens[0]), float(tokens[1])])
            features.append(cur_feature_list)
            labels.append(label)
            if cnt == batch_size:
                yield labels, features
                labels = []
                features = []
                cnt = 0
    if cnt > 0:
        yield labels, features

def prepare_data_4_sp(labels, features, dim):
    '''
        数据格式化以便序列化
    '''
    instance_cnt = len(labels)

    indices = [] # feature 序号
    values = [] # feature 值
    values_2 = [] # feature 值平方
    shape = [instance_cnt, dim] # 矩阵的shape
    feature_indices = [] # feature 序号

    for i in range(instance_cnt):
        m = len(features[i])
        for j in range(m):
            indices.append([i, features[i][j][0]])
            values.append(features[i][j][1])
            values_2.append(features[i][j][1] * features[i][j][1])
            feature_indices.append(features[i][j][0])

    res = {}

    res['indices'] = np.asarray(indices, dtype=np.int64)
    res['values'] = np.asarray(values, dtype=np.float32)
    res['values2'] = np.asarray(values_2, dtype=np.float32)
    res['shape'] = np.asarray(shape, dtype=np.int64)
    res['labels'] = np.asarray([[label] for label in labels], dtype=np.float32)
    res['feature_indices'] = np.asarray(feature_indices, dtype=np.int64)

    return res

def load_data_cache(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

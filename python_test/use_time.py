# coding:utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import time

# 用numpy创建一个 10x5 矩阵
# 加到默认图中.
from utils import MLP

graph_num = 50
graph_point_num = 4
point_dim = 3
time_step = 2# 小于graph_num
data_source = [[[i,i,i] for j in range(graph_point_num)] for i in range(graph_num)]
data_source = np.array(data_source)
data_source = data_source.reshape([-1,3])
#data_source = np.arange(graph_num*graph_point_num*point_dim).reshape(-1, point_dim)
#data_source = np.ones_like(data_source)
source_data = tf.constant(data_source, dtype = tf.float32)
source_data_extend = source_data
for i in range(time_step-1):
    source_data_extend = tf.concat([source_data[(i+1)*graph_point_num:(i+2)*graph_point_num,:], source_data_extend], 0)
source_data = source_data_extend
time_message_aggregation = tf.Variable(name = "time", initial_value = tf.ones([time_step])/time_step, dtype = tf.float32)

list_source_data = []

for i in range(time_step):
    if i ==time_step-1:
        list_source_data.append(source_data[i*graph_point_num:])
    else:
        list_source_data.append(source_data[i*graph_point_num:
                                            -((time_step -1)-i)*graph_point_num])

for id, one_list_source_data in enumerate(list_source_data):
    if id ==0:
        time_message_aggregation_source_data = one_list_source_data#*time_message_aggregation[id]
    else:
        #time_message_aggregation_source_data +=one_list_source_data*time_message_aggregation[id]
        time_message_aggregation_source_data = tf.concat([time_message_aggregation_source_data, one_list_source_data], 1)

# target = tf.reshape(target,shape = (-1,3))
# 启动默认图，执行这个乘法操作
with tf.Session() as sess:
    with sess.graph.as_default():
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

    a = (sess.run([source_data]))
    b = (sess.run([time_message_aggregation_source_data]))

    print(sess.run([source_data]))
    print(sess.run([time_message_aggregation_source_data]))
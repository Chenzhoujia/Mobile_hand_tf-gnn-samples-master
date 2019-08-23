# coding:utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import time

# 用numpy创建一个 10x5 矩阵
# 加到默认图中.
from utils import MLP
from utils.general import NetworkOps

ops = NetworkOps
def hand_move(joint_uvd, time_step = 1):
    joint_uvd_move = joint_uvd[time_step:]
    for i in range(time_step):
        joint_uvd_move = tf.concat([joint_uvd_move, joint_uvd_move[-(i+2)][np.newaxis, :, :]], 0)
    return joint_uvd_move

graph_num = 50
graph_point_num = 4
point_dim = 3
time_step = 5# 小于graph_num
data_source = [[[i,i,i] for j in range(graph_point_num)] for i in range(graph_num)]
data_source = np.array(data_source)
data_source = data_source.reshape([-1,3])
#data_source = np.arange(graph_num*graph_point_num*point_dim).reshape(-1, point_dim)
#data_source = np.ones_like(data_source)
source_data = tf.constant(data_source, dtype = tf.float32)
source_data_move = hand_move(tf.reshape(source_data,shape = (graph_num,graph_point_num,point_dim)))
source_data_move = tf.reshape(source_data_move, shape = (graph_num*graph_point_num,point_dim))
source_data_extend = source_data
for i in range(time_step-1):
    source_data_extend = tf.concat([source_data[(i+1)*graph_point_num:(i+2)*graph_point_num,:], source_data_extend], 0)

source_data = source_data_extend
time_message_aggregation = tf.Variable(name = "time", initial_value = tf.ones([time_step])/time_step, dtype = tf.float32)

shape = tf.shape(source_data)
source_data_sum = tf.reduce_sum(source_data,1)
targets_control_bool = tf.not_equal(source_data_sum, 0)
targets_control_bool = tf.expand_dims(targets_control_bool, -1)
targets_control_bool = tf.tile(targets_control_bool, (1,3))
targets_control_bool = tf.cast(targets_control_bool,tf.float32)

targets_control_bool1 = targets_control_bool*1.5
targets_control_bool2 = targets_control_bool*2.5
targets_control_bool_ = (1.0-targets_control_bool)
targets_control_bool3 = targets_control_bool1*targets_control_bool_

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

time_message_aggregation_source_data_reshape = tf.reshape(time_message_aggregation_source_data, shape = (-1,time_step,point_dim))

# 用叠加的卷积层代替 单层全连接
out_chan_list = [3]
for i, out_chan in enumerate(out_chan_list):
    time_message_aggregation_source_data_reshape = ops.conv1_relu6(time_message_aggregation_source_data_reshape, 'fc_vp_%d' % (i), kernel_size=2, stride =1, out_chan=out_chan, trainable=True)
time_message_aggregation_source_data_reshape = tf.reshape(time_message_aggregation_source_data_reshape, shape = (-1,(time_step-1)*3))

with tf.Session() as sess:
    with sess.graph.as_default():
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

    a = (sess.run(source_data))
    b = (sess.run(time_message_aggregation_source_data))
    c = (sess.run(time_message_aggregation_source_data_reshape))

    # print(sess.run(source_data))
    # print(sess.run(source_data_move))
    print(sess.run(time_message_aggregation_source_data))
    print(sess.run(time_message_aggregation_source_data_reshape))
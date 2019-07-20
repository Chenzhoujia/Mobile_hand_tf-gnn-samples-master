# coding:utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import time

# 用numpy创建一个 10x5 矩阵
# 加到默认图中.
from utils import MLP
source_data = tf.constant([[1,1,1,1], [2,2,2,2], [3,3,3,3],
                           [-1.1, -1.2, -1.3, -1.4], [-2.1, -2.2, -2.3, -2.4], [-3.1, -3.2, -3.3, -3.4]], dtype = tf.float32)
dense_message_aggregation = tf.Variable(name = "s", initial_value = tf.ones([3, 3]), dtype = tf.float32)
dense_message_aggregation = tf.constant([[1,1,1], [2,2,2], [3,3,3]], dtype = tf.float32)

#source_data = tf.reshape(source_data, shape=(-1,3,4))

source_data_tmp = tf.transpose(source_data) #[D, B*G]
source_data_tmp = tf.reshape(source_data_tmp,shape = (-1,3))#[D*B, G]
target = tf.matmul(source_data_tmp, dense_message_aggregation)#[D*B, G]
target = tf.reshape(target,shape = (4, -1))#[D, B*G]
target = tf.transpose(target) #[B*G, D]
target = tf.reshape(target,shape = (-1, 3, 4)) #[B ,G, D]

# target = tf.reshape(target,shape = (-1,3))
# 启动默认图，执行这个乘法操作
with tf.Session() as sess:
    with sess.graph.as_default():
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

    print(sess.run([source_data]))
    print(sess.run([dense_message_aggregation]))
    print(sess.run([source_data_tmp]))
    print(sess.run([target]))
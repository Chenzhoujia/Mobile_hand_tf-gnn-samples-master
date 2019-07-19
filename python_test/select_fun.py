# coding:utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import time

# 用numpy创建一个 10x5 矩阵
# 加到默认图中.
from utils import MLP
point_num = 32
select_point_num = 5
batch_num = np.random.randint(3,6)
source_data = tf.convert_to_tensor(np.random.random((point_num*batch_num, 3)), dtype = tf.float32)  # 用于计算相乘和点乘（内积）的矩阵
# [b*32,3]
select = tf.reshape( source_data, [-1, point_num, 3])# [b,32,3]
select = tf.reshape( select, [-1, point_num*3])# [b,32*3]
with tf.variable_scope("regression_gate"):
    regression_gate = \
        MLP(point_num*3, 100, [], 1)
with tf.variable_scope("regression"):
    regression_transform = \
        MLP(100, select_point_num, [], 1)

select = regression_gate(select)
select = regression_transform(select)
select = tf.minimum(tf.maximum(select, -1), 1)
select = (select + 1)/2*(point_num-1)
select = tf.reduce_mean(select, axis=0) # [b,5]
select = tf.round(select)

mask = tf.expand_dims(tf.range(point_num), 1)
mask = tf.cast(tf.tile(mask, [1, 3]), tf.float32)
ones_mask = tf.ones_like(mask)
zeros_mask = tf.zeros_like(mask)

for point in range(select_point_num):
    if point == 0:
        mask_log = tf.equal(mask - select[point], zeros_mask)
    else:
        mask_log =  tf.logical_or(mask_log, tf.equal(mask - select[point], zeros_mask))

select = tf.where(mask_log, ones_mask, zeros_mask)
source_data = tf.reshape(source_data,[-1, point_num, 3])
source_data = tf.multiply(source_data, select)
source_data = tf.reshape(source_data,[-1, 3])
#product_1 = tf.multiply(matrix1, select)  # 矩阵点乘（内积）
#mask -=select[1]
# 启动默认图，执行这个乘法操作
with tf.Session() as sess:
    with sess.graph.as_default():
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

    print(sess.run([select]))
    print(sess.run([source_data]))
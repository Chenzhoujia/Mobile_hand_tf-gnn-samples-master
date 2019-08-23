# coding:utf-8
import tensorflow as tf
import numpy as np

# 定义一个矩阵a，表示需要被卷积的矩阵。
a = np.array(np.arange(1, 1 + 20).reshape([1, 10, 2]), dtype=np.float32)

# 卷积核，此处卷积核的数目为1
kernel = np.array(np.arange(1, 1 + 4), dtype=np.float32).reshape([2, 2, 1])

"""
1、value：在注释中，value的格式为：[batch, in_width, in_channels]，batch为样本维，表示多少个样本，in_width为宽度维，表示样本的宽度，in_channels维通道维，表示样本有多少个通道。 
  事实上，也可以把格式看作如下:[batch, 行数, 列数]，把每一个样本看作一个平铺开的二维数组。这样的话可以方便理解。

2、filters：在注释中，filters的格式为：[filter_width, in_channels, out_channels]。按照value的第二种看法，filter_width可以看作每次与value进行卷积的行数，in_channels表示value一共有多少列（与value中的in_channels相对应）。out_channels表示输出通道，可以理解为一共有多少个卷积核，即卷积核的数目。

3、stride：一个整数，表示步长，每次（向下）移动的距离（TensorFlow中解释是向右移动的距离，这里可以看作向下移动的距离）。

4、padding：同conv2d，value是否需要在下方填补0。

5、name：名称。可省略。
 ———————————————— 
版权声明：本文为CSDN博主「子为空」的原创文章，遵循CC 4.0 by-sa版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/DaVinciL/article/details/81359245
"""

# 进行conv1d卷积
conv1d = tf.nn.conv1d(a, kernel, 1, 'VALID')

with tf.Session() as sess:
    # 初始化
    tf.global_variables_initializer().run()
    # 输出卷积值
    print(a)
    print(kernel)
    print(sess.run(conv1d))
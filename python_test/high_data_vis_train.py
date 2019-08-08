from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os


# 简单卷积层，为方便本章教程叙述，固定部分参数
def conv_layer(input,
               channels_in,  # 输入通道数
               channels_out,  # 输出通道数
               name='conv'):  # 名称
    with tf.name_scope(name):
        weights = tf.Variable(tf.truncated_normal([5, 5, channels_in, channels_out],
                                                  stddev=0.05), name='W')
        biases = tf.Variable(tf.constant(0.05, shape=[channels_out]), name='B')
        conv = tf.nn.conv2d(input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
        act = tf.nn.relu(conv + biases)

        tf.summary.histogram('weights', weights)
        tf.summary.histogram('biases', biases)
        tf.summary.histogram('activations', act)

        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 简化全连接层
def fc_layer(input, num_inputs, num_outputs, name='fc'):
    with tf.name_scope(name):
        weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs],
                                                  stddev=0.05), name='W')
        biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]), name='B')
        act = tf.matmul(input, weights) + biases

        tf.summary.histogram('weights', weights)
        tf.summary.histogram('biases', biases)
        tf.summary.histogram('activations', act)

        return act


def max_pool(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


from tensorflow.examples.tutorials.mnist import input_data

"""
LABELS  标签
SPRITES 输入图片的集合
embedding 就是将离散目标到高维特征的这个mapping


https://distill.pub/2016/misread-tsne/ tsne的解读
Tsne：目标是在高维空间中获取一组点，并在较低维空间（通常是2D平面）中找到这些点的表示。
SNE构建一个高维对象之间的概率分布，使得相似的对象有更高的概率被选择，而不相似的对象有较低的概率被选择。
SNE在低维空间里在构建这些点的概率分布，使得这两个概率分布之间尽可能的相似。

（1）该算法是非线性的，并且适应底层数据，在不同区域上执行不同的变换
具有超参数 perplexity 说明如何在数据的本地和全局方面之间平衡注意力。在某种意义上，该参数是关于每个点具有的近邻的数量的猜测。
较小时，局部特征主导，否则全局特征主导
（2）step表示的是sne算法执行的次数，而非网络train的次数，不同的数据集可能需要不同的迭代次数才能收敛。
具有相同超参数的不同运行，不保证相同的结果
（3）簇的包围盒的大小毫无意义， t-SNE算法使其“距离”概念适应数据集中的区域密度变化。 结果，它自然地扩展了密集的集群，并且收缩了稀疏集群
（4）簇之间的距离毫无意义，会受到perplexity的影响
（5）当perplexity很小时，会在噪音中看到一些聚类，这是没有意义的，通过遍历perplexity，可以观察到一些有意义的稳定的结构


http://setosa.io/ev/principal-component-analysis/ PCA的解读

PCA就是通过坐标轴的变换，选择最有代表性的几个轴
"""
#
# LABELS = os.path.join(os.getcwd(), "tf-dev-summit-tensorboard-tutorial-master/labels_1024.tsv")
# SPRITES = os.path.join(os.getcwd(), "tf-dev-summit-tensorboard-tutorial-master/sprite_1024.png")
LABELS = os.path.join(os.getcwd(), "tf-dev-summit-tensorboard-tutorial-master/mnist_meta.tsv")
SPRITES = os.path.join(os.getcwd(), "tf-dev-summit-tensorboard-tutorial-master/mnist_sprite.jpg")

def mnist_model(learning_rate, use_two_fc, use_two_conv, hparam):
    tf.reset_default_graph()  # 重置计算图
    sess = tf.Session()

    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 3)
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

    if use_two_conv:  # 是否使用两个卷积
        conv1 = conv_layer(x_image, 1, 32, "conv1")
        conv_out = conv_layer(conv1, 32, 64, "conv2")
    else:
        conv1 = conv_layer(x_image, 1, 64, "conv")  # 如果使用一个卷积，则再添加一个max_pooling保证维度相通
        conv_out = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])

    if use_two_fc:  # 是否使用两个全连接
        fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, "fc1")
        relu = tf.nn.relu(fc1)
        embedding_input = relu
        tf.summary.histogram("fc1/relu", relu)
        embedding_size = 1024
        logits = fc_layer(fc1, 1024, 10, "fc2")
    else:
        embedding_input = flattened  # 新添加的embedding_input和embedding_size
        embedding_size = 7 * 7 * 64
        logits = fc_layer(flattened, 7 * 7 * 64, 10, "fc")

    with tf.name_scope("xent"):
        xent = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=y), name="xent")
        tf.summary.scalar("xent", xent)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    summ = tf.summary.merge_all()  # 收集所有的summary

    # 添加embedding变量
    embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
    assignment = embedding.assign(embedding_input)
    saver = tf.train.Saver()  # 保存训练过程

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(tensorboard_dir + hparam)
    writer.add_graph(sess.graph)

    # embedding的配置，详见官方文档
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.sprite.image_path = SPRITES
    embedding_config.metadata_path = LABELS
    # Specify the width and height of a single thumbnail.
    embedding_config.sprite.single_image_dim.extend([28, 28])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

    for i in range(2001):
        batch = data.train.next_batch(100)
        if i % 5 == 0:  # 每5轮写入一次
            # 同样，最好使用验证集
            [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y: batch[1]})
            writer.add_summary(s, i)

        if i % 100 == 0:  # 每100轮保存依存训练过程
            sess.run(assignment, feed_dict={x: data.test.images[:1024], y: data.test.labels[:1024]})
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1]})
            saver.save(sess, os.path.join(tensorboard_dir, "model.ckpt"), i)

            print("迭代轮次: {0:>6}, 训练准确率: {1:>6.4%}".format(i, train_accuracy))

        sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
    conv_param = "conv=2" if use_two_conv else "conv=1"
    fc_param = "fc=2" if use_two_fc else "fc=1"
    return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)

tensorboard_dir = 'tensorboard/mnist4/'   # 保存目录
data = input_data.read_data_sets('data/MNIST', one_hot=True)

if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)

for learning_rate in [1E-3, 1E-4, 1e-5]:
    for use_two_fc in [False, True]:
        for use_two_conv in [False, True]:
            hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
            print('Starting run for %s' % hparam)

            mnist_model(learning_rate, use_two_fc, use_two_conv, hparam)

print('Done training!')
# -*- coding: utf-8 -*-
import datetime
import tensorflow as tf
import os
import platform
import time
import numpy as np
import configparser

from tqdm import tqdm

# from dataset_interface.RHD import RHD
from data.BinaryDbReader import BinaryDbReader
# from dataset_interface.dataset_prepare import CocoPose
# from src.networks import get_network
import matplotlib.pyplot as plt
from utils.general import NetworkOps
# from src import  network_mv2_hourglass
from mpl_toolkits.mplot3d import Axes3D
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
ops = NetworkOps
def upsample(inputs, factor, name):
    return tf.image.resize_bilinear(inputs, [int(inputs.get_shape()[1]) * factor, int(inputs.get_shape()[2]) * factor],
                                    name=name)
def get_loss_and_output(scoremap, pose):
    # 叠加在batch上重用特征提取网络
    keypoints_scoremap = scoremap
    train=True
    bottleneck=False
    num_kp = 32
    """ Inference of canonical coordinates. """
    with tf.variable_scope('PosePrior'):
        # use encoding to detect relative, normed 3d coords
        x = keypoints_scoremap  # this is 28x28x21
        s = x.get_shape().as_list()
        out_chan_list = [32, 64, 128]
        for i, out_chan in enumerate(out_chan_list):
            x = ops.conv_relu(x, 'conv_pose_%d_1' % i, kernel_size=3, stride=1, out_chan=out_chan, trainable=train)
            x = ops.conv_relu(x, 'conv_pose_%d_2' % i, kernel_size=3, stride=2, out_chan=out_chan, trainable=train) # in the end this will be 4x4xC

        # Estimate relative 3D coordinates
        out_chan_list = [512, 512]
        x = tf.reshape(x, [s[0], -1])

        for i, out_chan in enumerate(out_chan_list):
            x = ops.fully_connected_relu(x, 'fc_rel%d' % i, out_chan=out_chan, trainable=train)
        if bottleneck:
            x = ops.fully_connected(x, 'fc_bottleneck', out_chan=30)
        coord_xyz_rel = ops.fully_connected(x, 'fc_xyz', out_chan=num_kp*3, trainable=train)

        # reshape stuff
        coord_xyz_rel = tf.reshape(coord_xyz_rel, [s[0], num_kp, 3])
        per_graph_errors_per = tf.square(tf.cast(coord_xyz_rel, tf.float32) - tf.cast(pose, tf.float32))
        per_graph_errors_per = tf.reduce_mean(per_graph_errors_per, axis=-1)
        per_graph_errors_per = tf.reduce_sum(tf.reduce_mean(per_graph_errors_per, axis=0))

        return per_graph_errors_per, coord_xyz_rel

def average_gradients(tower_grads):
    """
    Get gradients of all variables.
    :param tower_grads:
    :return:
    """
    average_grads = []

    # get variable and gradients in differents gpus
    for grad_and_vars in zip(*tower_grads):
        # calculate the average gradient of each gpu
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def main(argv=None):
    # load config file and setup
    params = {}
    params['visible_devices'] = '0'
    params['modelpath'] = './data/hand_gen/rich/point/infer_z_CNN/model'
    params['logpath'] = './data/hand_gen/rich/point/infer_z_CNN/log'
    params['lr'] = '0.001'
    params['decay_rate'] = '0.95'
    os.environ['CUDA_VISIBLE_DEVICES'] = params['visible_devices']
    params['batchsize'] = 64
    params['gpus'] = 1
    params['size'] = 40
    params['num_key'] = 32
    params['max_epoch'] = 1000
    params['per_saved_model_step'] = 500
    params['per_update_tensorboard_step'] = 500

    if not os.path.exists(params['modelpath']):
        os.makedirs(params['modelpath'])
    if not os.path.exists(params['logpath']):
        os.makedirs(params['logpath'])

    gpus = 'gpus'

    if platform.system() == 'Darwin':
        gpus = 'cpu'
    training_name = 'infer_z_CNN'
    evaluation = tf.placeholder_with_default(True, shape=())
    with tf.Graph().as_default(), tf.device("/cpu:0"):
        dataset_RHD = BinaryDbReader(batch_size=params['batchsize'],mode = 'training')

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(float(params['lr']), global_step,
                                                   decay_steps=10000, decay_rate=float(params['decay_rate']),
                                                   staircase=True)
        opt = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
        tower_grads = []

        for i in range(params['gpus']):
            with tf.device("/gpu:%d" % i):
                with tf.name_scope("GPU_%d" % i):
                    #input_image, keypoint_xyz, keypoint_uv, input_heat, keypoint_vis, k, num_px_left_hand, num_px_right_hand \
                    batch_data_all = dataset_RHD.get_batch_data
                    scoremap = batch_data_all[0]
                    scoremap.set_shape((params['batchsize'], params['size'],params['size'], params['num_key']))
                    pose = batch_data_all[1]
                    pose.set_shape((params['batchsize'], params['num_key'], 3))
                    loss, preheat= get_loss_and_output(scoremap, pose)

                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)

        grads = average_gradients(tower_grads)
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram("gradients_on_average/%s" % var.op.name, grad)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        MOVING_AVERAGE_DECAY = 0.99
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_to_average = (tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variable_to_average)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.group(apply_gradient_op, variables_averages_op)

        saver = tf.train.Saver(max_to_keep=10)

        tf.summary.scalar("learning_rate", learning_rate)
        tf.summary.scalar("loss", loss)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        # occupy gpu gracefully
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            init.run()
            checkpoint_path = os.path.join(params['modelpath'], training_name)
            model_name = '/model-2000'
            if checkpoint_path:
                saver.restore(sess, checkpoint_path+model_name)
                print("restore from " + checkpoint_path+model_name)
            total_step_num = 196443 * params['max_epoch'] // (params['batchsize']* 2 * params['gpus'])

            print("Start training...")
            for step in tqdm(range(total_step_num)):
                _, loss_value = sess.run([train_op, loss])
                if step % params['per_update_tensorboard_step'] == 0:
                    # [total_loss, loss_scoremap, loss_zrate, z_rate_pre]
                    loss_v, preheat_v, pose_v= sess.run([loss, preheat, pose])
                    preheat_v = preheat_v[0]
                    pose_v = pose_v[0]
                    fig = plt.figure(1)
                    plt.clf()

                    for draw_i in range(2):
                        if draw_i==0:
                            pose_show = preheat_v
                            ax = fig.add_subplot(121, projection='3d')
                        else:
                            pose_show = pose_v
                            ax = fig.add_subplot(122, projection='3d')

                        ax.view_init(azim=45.0, elev=20.0)  # aligns the 3d coord with the camera view
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                        ax.set_zlabel('z')
                        ax.set_xlim((-1, 1))
                        ax.set_ylim((-1, 1))
                        ax.set_zlim((-1, 1))
                        fig_color = ['c', 'm', 'y', 'g', 'r', 'b']
                        for f in range(6):
                            if f < 5:
                                for bone in range(5):
                                    ax.plot([pose_show[f * 6 + bone, 0], pose_show[f * 6 + bone + 1, 0]],
                                            [pose_show[f * 6 + bone, 1], pose_show[f * 6 + bone + 1, 1]],
                                            [pose_show[f * 6 + bone, 2], pose_show[f * 6 + bone + 1, 2]], color=fig_color[f],
                                            linewidth=2)
                                if f == 4:
                                    ax.plot([pose_show[f * 6 + bone + 1, 0], pose_show[30, 0]],
                                            [pose_show[f * 6 + bone + 1, 1], pose_show[30, 1]],
                                            [pose_show[f * 6 + bone + 1, 2], pose_show[30, 2]], color=fig_color[f], linewidth=2)
                                else:
                                    ax.plot([pose_show[f * 6 + bone + 1, 0], pose_show[31, 0]],
                                            [pose_show[f * 6 + bone + 1, 1], pose_show[31, 1]],
                                            [pose_show[f * 6 + bone + 1, 2], pose_show[31, 2]], color=fig_color[f], linewidth=2)
                            # ax.scatter(uvd_pt[f * 2, 0], uvd_pt[f * 2, 1], uvd_pt[f * 2, 2], s=60, c=fig_color[f])
                            ax.scatter(pose_show[f * 6:(f + 1) * 6, 0], pose_show[f * 6:(f + 1) * 6, 1], pose_show[f * 6:(f + 1) * 6, 2],
                                       s=30, c=fig_color[f])

                    plt.savefig(os.path.join(params['logpath']) + "/" + str(step).zfill(10) + "_.png")

                    print("loss:"+str(loss_v))

                    # save model
                if step % params['per_saved_model_step'] == 0:
                    saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=step)

if __name__ == '__main__':
    tf.app.run()

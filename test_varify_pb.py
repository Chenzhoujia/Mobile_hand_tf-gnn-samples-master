import tensorflow as tf
import argparse
import cv2
from tqdm import tqdm
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import pickle, os
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from typing import Any, Dict, Tuple, List, Iterable

test_model = 'rich/'
level_model = 'point/'
detal_name = 'smart_motion_hand/'
methon_name = 'tip_control'
detal_name += methon_name + '/'
save_dataset_dir = "data/hand_gen/"+test_model+level_model+detal_name
time_agg_step = 15

def load_graph(frozen_graph_filename):
    # 加载protobug文件，并反序列化成graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph_:
        # 将读出来的graph_def导入到当前的Graph中
        # 为了避免多个图之间的明明冲突，增加一个前缀
        tf.import_graph_def(graph_def)

    return graph_

def load_data(data_file):
    print(" Loading hand data from %s." % (data_file,))
    data_file = "%s" % (data_file,)
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data
def draw_3d_point_all(inputs, outputs, labels, step): #

    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')

    bone_len = []
    bone_fake_len = []
    for draw_i in range(3):
        if draw_i == 0:
            pose_show = inputs
            fig_color = ['r', 'r', 'r', 'r', 'r', 'r']
        elif draw_i == 1:
            pose_show = outputs
            fig_color = ['k', 'k', 'k', 'k', 'k', 'k']
        else:
            pose_show = labels
            fig_color = ['b', 'b', 'b', 'b', 'b', 'b']
        ax.view_init(azim=20.0, elev=40.0)  # aligns the 3d coord with the camera view
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        """
        蓝色： 'b' (blue)
        绿色： 'g' (green)
        红色： 'r' (red)
        蓝绿色(墨绿色)： 'c' (cyan)
        红紫色(洋红)： 'm' (magenta)
        黄色： 'y' (yellow)
        黑色： 'k' (black)
        白色： 'w' (white)
        """
        #fig_color = ['c', 'm', 'y', 'g', 'r', 'b']
        for f in range(6):
            if f < 5:
                for bone in range(5):
                    ax.plot([pose_show[f * 6 + bone, 0], pose_show[f * 6 + bone + 1, 0]],
                            [pose_show[f * 6 + bone, 1], pose_show[f * 6 + bone + 1, 1]],
                            [pose_show[f * 6 + bone, 2], pose_show[f * 6 + bone + 1, 2]], color=fig_color[f], linewidth=0.5)
                if f == 4:
                    ax.plot([pose_show[f * 6 + bone + 1, 0], pose_show[30, 0]],
                            [pose_show[f * 6 + bone + 1, 1], pose_show[30, 1]],
                            [pose_show[f * 6 + bone + 1, 2], pose_show[30, 2]], color=fig_color[f], linewidth=0.5)
                else:
                    ax.plot([pose_show[f * 6 + bone + 1, 0], pose_show[31, 0]],
                            [pose_show[f * 6 + bone + 1, 1], pose_show[31, 1]],
                            [pose_show[f * 6 + bone + 1, 2], pose_show[31, 2]], color=fig_color[f], linewidth=0.5)
            ax.scatter(pose_show[f * 6:(f + 1) * 6, 0], pose_show[f * 6:(f + 1) * 6, 1], pose_show[f * 6:(f + 1) * 6, 2], s=3,
                       c=fig_color[f])


    ax.set_title(str(bone_len)+"\n"+str(bone_fake_len),fontsize=10)
    # plt.show()
    # plt.pause(0.01)
    directory = save_dataset_dir+"trained_model/test/image/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + "pb_" + str(step).zfill(5) + ".jpg")
def draw_3d_point_IO(inputs, outputs, step): #

    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')

    bone_len = []
    bone_fake_len = []
    for draw_i in range(2):
        if draw_i == 0:
            pose_show = inputs
            fig_color = ['r', 'r', 'r', 'r', 'r', 'r']
        else :
            pose_show = outputs
            fig_color = ['k', 'k', 'k', 'k', 'k', 'k']
        ax.view_init(azim=20.0, elev=40.0)  # aligns the 3d coord with the camera view
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim((-2, 1))
        ax.set_ylim((-1.5, 1.5))
        ax.set_zlim((1, 5))
        """
        蓝色： 'b' (blue)
        绿色： 'g' (green)
        红色： 'r' (red)
        蓝绿色(墨绿色)： 'c' (cyan)
        红紫色(洋红)： 'm' (magenta)
        黄色： 'y' (yellow)
        黑色： 'k' (black)
        白色： 'w' (white)
        """
        #fig_color = ['c', 'm', 'y', 'g', 'r', 'b']
        for f in range(6):
            if f < 5:
                for bone in range(5):
                    ax.plot([pose_show[f * 6 + bone, 0], pose_show[f * 6 + bone + 1, 0]],
                            [pose_show[f * 6 + bone, 1], pose_show[f * 6 + bone + 1, 1]],
                            [pose_show[f * 6 + bone, 2], pose_show[f * 6 + bone + 1, 2]], color=fig_color[f], linewidth=2)
                if f == 4:
                    ax.plot([pose_show[f * 6 + bone + 1, 0], pose_show[30, 0]],
                            [pose_show[f * 6 + bone + 1, 1], pose_show[30, 1]],
                            [pose_show[f * 6 + bone + 1, 2], pose_show[30, 2]], color=fig_color[f], linewidth=2)
                else:
                    ax.plot([pose_show[f * 6 + bone + 1, 0], pose_show[31, 0]],
                            [pose_show[f * 6 + bone + 1, 1], pose_show[31, 1]],
                            [pose_show[f * 6 + bone + 1, 2], pose_show[31, 2]], color=fig_color[f], linewidth=2)
            ax.scatter(pose_show[f * 6:(f + 1) * 6, 0], pose_show[f * 6:(f + 1) * 6, 1], pose_show[f * 6:(f + 1) * 6, 2], s=30,
                       c=fig_color[f])


    ax.set_title(str(bone_len)+"\n"+str(bone_fake_len),fontsize=10)
    # plt.show()
    # plt.pause(0.01)
    directory = save_dataset_dir+"trained_model/test/image/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + "pb_" + str(step).zfill(5) + ".jpg")

def draw_3d_point_control(inputs, outputs, step): #

    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')

    bone_len = []
    bone_fake_len = []
    for draw_i in range(2):
        if draw_i == 0:
            pose_show = inputs
            fig_color = ['r', 'r', 'r', 'r', 'r', 'r']
        else :
            pose_show = outputs
            fig_color = ['k', 'k', 'k', 'k', 'k', 'k']
        ax.view_init(azim=20.0, elev=40.0)  # aligns the 3d coord with the camera view
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim((-2, 1))
        ax.set_ylim((-1.5, 1.5))
        ax.set_zlim((1, 5))
        """
        蓝色： 'b' (blue)
        绿色： 'g' (green)
        红色： 'r' (red)
        蓝绿色(墨绿色)： 'c' (cyan)
        红紫色(洋红)： 'm' (magenta)
        黄色： 'y' (yellow)
        黑色： 'k' (black)
        白色： 'w' (white)
        """
        #fig_color = ['c', 'm', 'y', 'g', 'r', 'b']
        for f in range(6):
            if f < 5:
                for bone in range(5):
                    ax.plot([pose_show[f * 6 + bone, 0], pose_show[f * 6 + bone + 1, 0]],
                            [pose_show[f * 6 + bone, 1], pose_show[f * 6 + bone + 1, 1]],
                            [pose_show[f * 6 + bone, 2], pose_show[f * 6 + bone + 1, 2]], color=fig_color[f], linewidth=2)
                if f == 4:
                    ax.plot([pose_show[f * 6 + bone + 1, 0], pose_show[30, 0]],
                            [pose_show[f * 6 + bone + 1, 1], pose_show[30, 1]],
                            [pose_show[f * 6 + bone + 1, 2], pose_show[30, 2]], color=fig_color[f], linewidth=2)
                else:
                    ax.plot([pose_show[f * 6 + bone + 1, 0], pose_show[31, 0]],
                            [pose_show[f * 6 + bone + 1, 1], pose_show[31, 1]],
                            [pose_show[f * 6 + bone + 1, 2], pose_show[31, 2]], color=fig_color[f], linewidth=2)
            ax.scatter(pose_show[f * 6:(f + 1) * 6, 0], pose_show[f * 6:(f + 1) * 6, 1], pose_show[f * 6:(f + 1) * 6, 2], s=30,
                       c=fig_color[f])


    ax.set_title(str(bone_len)+"\n"+str(bone_fake_len),fontsize=10)
    # plt.show()
    # plt.pause(0.01)
    directory = save_dataset_dir+"trained_model/test/image/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + "pb_" + str(step).zfill(5) + ".jpg")

if __name__ == '__main__':

    # 允许用户传入文件名作为参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="results/frozen_model.pb", type=str,
                        help="Frozen model file to import")
    args = parser.parse_args()

    # 从pb文件中读取图结构
    model_path = save_dataset_dir + "trained_model/HAND_GEN_GGNN_2019-08-23-16-24-37_4788_best_model.pb"
    graph = load_graph(model_path)

    # 列举所有的操作
    for op in graph.get_operations():
        print(op.name)
        # --input_arrays=initial_node_features \
        # --output_arrays=out_layer_task0/final_output_node_representations \
    x = graph.get_tensor_by_name('import/initial_node_features:0')
    x2 = graph.get_tensor_by_name('import/targets_control:0')
    y = graph.get_tensor_by_name('import/out_layer_task0/final_output_node_representations:0')

    # 读取数据
    data = load_data(save_dataset_dir+"hand_test.pkl")
    # 展示所有图片
    # for data_i in range(len(data)):
    #     input_v = data[data_i]['node_features']
    #     preheat_v = data[data_i]['targets']
    #     draw_3d_point_IO(input_v, preheat_v, data_i)
    # We launch a Session
    with tf.Session(graph=graph) as sess:
        # for step in tqdm(range(len(data)-time_agg_step)):
        #     input = []
        #     target = []
        #     control = []
        #     for time in range(time_agg_step):
        #         input.extend(data[step+time]['node_features'])
        #         target.extend(data[step+time]['targets'])
        #         control.extend(data[step+time]['targets_control'])
        #     input = np.array(input)
        #     target = np.array(target)
        #     control = np.array(control)
        #     preheat_v = sess.run(y, feed_dict={x: input, x2:control})
        #     draw_3d_point_all(input[-36:], preheat_v[-36:], target[-36:], step)
        # 选择一个开始， 不用label了 位置随机 步长随机
        start = 260-15
        step = 0
        input = []
        target = []
        control = []
        for step in range(time_agg_step):
            input.extend(data[step + start]['node_features'])
            target.extend(data[step + start]['targets'])
            control.extend(data[step + start]['targets_control'])
        input = np.array(input)
        target = np.array(target)
        control = np.array(control)
        # 开始循环
        # 先热一下身
        preheat_v = sess.run(y, feed_dict={x: input, x2:control})
        draw_3d_point_IO(input[-36:], preheat_v[-36:], step)
        # 加载 控制轨迹
        draw_control = np.load(save_dataset_dir + "test2019-08-28 17:19:09.npy")

        while(1):
            preheat_v = sess.run(y, feed_dict={{x: input, x2:control}})
            draw_3d_point_IO(input[-36:], preheat_v[-36:], step)
            input = input[36:]
            preheat_v = preheat_v[-36:]
            input = np.concatenate((input, preheat_v), axis=0)
            step = step+1

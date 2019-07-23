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

def load_data(data_file = './/data/hand_gen/hand_test.pkl'):
    print(" Loading hand data from %s." % (data_file,))
    data_file = "%s" % (data_file,)
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data
def draw(input, target, result, select, step):
    #观察序列，查看关键点坐标，确定角度由哪些坐标计算
    fig = plt.figure(1)
    fig_color = ['c', 'm', 'y', 'g', 'r', 'b']
    fig.clear()
    ax1 = fig.add_subplot(221, projection='3d')
    for select_i in range(32):
        select_tmp_c = int(select_i/6)
        ax1.scatter(input[select_i, 0], input[select_i, 1],input[select_i, 2], c=fig_color[select_tmp_c])
    ax1.view_init(azim=45.0, elev=20.0)  # aligns the 3d coord with the camera view
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_xlim((-1, 1))
    ax1.set_ylim((-1, 1))
    ax1.set_zlim((-1, 1))

    ax2 = fig.add_subplot(222, projection='3d')
    ax2.view_init(azim=45.0, elev=20.0)  # aligns the 3d coord with the camera view
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_xlim((-1, 1))
    ax2.set_ylim((-1, 1))
    ax2.set_zlim((-1, 1))

    for f in range(6):
        if f < 5:
            for bone in range(5):
                ax2.plot([target[f * 6+bone, 0], target[f * 6 +bone+ 1, 0]],
                        [target[f * 6+bone, 1], target[f * 6 +bone+ 1, 1]],
                        [target[f * 6+bone, 2], target[f * 6 +bone+ 1, 2]], color=fig_color[f], linewidth=2)
            if f == 4:
                ax2.plot([target[f * 6 + bone + 1, 0], target[30, 0]],
                        [target[f * 6 + bone + 1, 1], target[30, 1]],
                        [target[f * 6 + bone + 1, 2], target[30, 2]], color=fig_color[f], linewidth=2)
            else:
                ax2.plot([target[f * 6 + bone + 1, 0], target[31, 0]],
                        [target[f * 6 + bone + 1, 1], target[31, 1]],
                        [target[f * 6 + bone + 1, 2], target[31, 2]], color=fig_color[f], linewidth=2)
        # ax.scatter(uvd_pt[f * 2, 0], uvd_pt[f * 2, 1], uvd_pt[f * 2, 2], s=60, c=fig_color[f])
        ax2.scatter(target[f*6:(f+1)*6, 0], target[f*6:(f+1)*6, 1], target[f*6:(f+1)*6, 2], s=30, c=fig_color[f])

    ax3 = fig.add_subplot(223, projection='3d')
    select = select.astype(np.int)
    for select_i in range(select.size):
        select_tmp = select[select_i]
        select_tmp_c = int(select_tmp/6)
        ax3.scatter(input[select[select_i], 0], input[select[select_i], 1],input[select[select_i], 2], c=fig_color[select_tmp_c])

    ax3.view_init(azim=45.0, elev=20.0)  # aligns the 3d coord with the camera view
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.set_xlim((-1, 1))
    ax3.set_ylim((-1, 1))
    ax3.set_zlim((-1, 1))
    ax3.set_title(str(select))

    ax3 = fig.add_subplot(224, projection='3d')
    ax2 = ax3
    target = result
    ax2.view_init(azim=45.0, elev=20.0)  # aligns the 3d coord with the camera view
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_xlim((-1, 1))
    ax2.set_ylim((-1, 1))
    ax2.set_zlim((-1, 1))
    fig_color = ['c', 'm', 'y', 'g', 'r', 'b']
    for f in range(6):
        if f < 5:
            for bone in range(5):
                ax2.plot([target[f * 6+bone, 0], target[f * 6 +bone+ 1, 0]],
                        [target[f * 6+bone, 1], target[f * 6 +bone+ 1, 1]],
                        [target[f * 6+bone, 2], target[f * 6 +bone+ 1, 2]], color=fig_color[f], linewidth=2)
            if f == 4:
                ax2.plot([target[f * 6 + bone + 1, 0], target[30, 0]],
                        [target[f * 6 + bone + 1, 1], target[30, 1]],
                        [target[f * 6 + bone + 1, 2], target[30, 2]], color=fig_color[f], linewidth=2)
            else:
                ax2.plot([target[f * 6 + bone + 1, 0], target[31, 0]],
                        [target[f * 6 + bone + 1, 1], target[31, 1]],
                        [target[f * 6 + bone + 1, 2], target[31, 2]], color=fig_color[f], linewidth=2)
        # ax.scatter(uvd_pt[f * 2, 0], uvd_pt[f * 2, 1], uvd_pt[f * 2, 2], s=60, c=fig_color[f])
        ax2.scatter(target[f*6:(f+1)*6, 0], target[f*6:(f+1)*6, 1], target[f*6:(f+1)*6, 2], s=30, c=fig_color[f])

    # plt.show()
    # plt.pause(0.01)
    directory = "/tmp/image/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + str(step).zfill(5) + "_.jpg")

if __name__ == '__main__':

    # 允许用户传入文件名作为参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="results/frozen_model.pb", type=str,
                        help="Frozen model file to import")
    args = parser.parse_args()

    # 从pb文件中读取图结构
    graph = load_graph(args.frozen_model_filename)

    # 列举所有的操作
    for op in graph.get_operations():
        print(op.name)
        # --input_arrays=initial_node_features \
        # --output_arrays=out_layer_task0/final_output_node_representations \
    x = graph.get_tensor_by_name('import/initial_node_features:0')
    y = graph.get_tensor_by_name('import/out_layer_task0/final_output_node_representations:0')

    # 读取数据
    data = load_data()

    # We launch a Session
    with tf.Session(graph=graph) as sess:
        for step in tqdm(range(len(data))):
            hand_data = np.loadtxt("./trained_models/test_image.txt", delimiter=',')
            hand_data = np.reshape(hand_data,(-1,3))
            #preheat_v = sess.run(y, feed_dict={x: data[step]['node_features']})
            preheat_v = sess.run(y, feed_dict={x: hand_data})
            #data_save = np.reshape(data[step]['node_features'],(1,-1))
            #np.savetxt("./trained_models/test_image.txt", data_save, fmt='%f', delimiter=',')
            draw(data[step]['node_features'], preheat_v, preheat_v, np.array([0,6,12,18,24,30,31]), step)
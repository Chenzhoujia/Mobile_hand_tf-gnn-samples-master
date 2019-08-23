#!/usr/bin/env python
"""
Usage:
   test.py [options] STORED_MODEL_PATH [DATA_PATH]

STORED_MODEL is the path of a model snapshot created by train.py.
DATA_PATH is the location of the data to test on.

Options:
    -h --help                       Show this screen.
    --result_dir DIR                Directory to store logfiles and trained models. [default: trained_models]
    --quiet                         Show less output.
    --debug                         Turn on debugger.
"""
from typing import Optional
import numpy as np
from docopt import docopt
import os
from dpu_utils.utils import run_and_debug, RichPath
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

from utils.model_utils import restore

# test_model = 'rich/'
# level_model = 'point/'
# detal_name = 'infer_z/'
# save_dataset_dir = "data/hand_gen/"+test_model+level_model+detal_name

test_model = 'rich/'
level_model = 'point/'
detal_name = 'smart_motion_hand/'
methon_name = 'tip_control'
detal_name += methon_name + '/'
save_dataset_dir = "data/hand_gen/"+test_model+level_model+detal_name

node_num = int(36)

def draw_3d_point(inputs, outputs, labels, step): #
    #['Palm', 'Wrist1', 'Wrist2', 'Thumb.R1', 'Thumb.R2', 'Thumb.T', 'Index.R', 'Index.T', 'Mid.R', 'Mid.T', 'Ring.R', 'Ring.T', 'Pinky.R', 'Pinky.T', 'Mean']
    label_id = [0,0,1,1,2,2,3,3,4,4,4,5,5,5]

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
            cur_id = [i for i, x in enumerate(label_id) if x == f] # 属于当前类别的下标

            for point_id in cur_id:
                ax.scatter(pose_show[point_id, 0], pose_show[point_id, 1],pose_show[point_id, 2], s=30, c=fig_color[f])

                if point_id!=cur_id[-1]:
                    ax.plot([pose_show[point_id, 0], pose_show[point_id + 1, 0]],
                            [pose_show[point_id, 1], pose_show[point_id + 1, 1]],
                            [pose_show[point_id, 2], pose_show[point_id + 1, 2]], color=fig_color[f],
                            linewidth=2)
                    len = np.sqrt(np.sum((pose_show[point_id] - pose_show[point_id + 1]) ** 2))
                    if draw_i == 0:
                        bone_len.append(round(len,1))
                    else:
                        bone_fake_len.append(round(len,1))

                # concat
                if point_id==cur_id[-1]:
                    ax.plot([pose_show[point_id, 0], pose_show[13, 0]],
                            [pose_show[point_id, 1], pose_show[13, 1]],
                            [pose_show[point_id, 2], pose_show[13, 2]], color=fig_color[f],
                            linewidth=2)
                    len = np.sqrt(np.sum((pose_show[point_id] - pose_show[13]) ** 2))
                    if draw_i == 0:
                        bone_len.append(round(len,1))
                    else:
                        bone_fake_len.append(round(len,1))


    ax.set_title(str(bone_len)+"\n"+str(bone_fake_len),fontsize=10)
    # plt.show()
    # plt.pause(0.01)
    directory = save_dataset_dir+"trained_model/test/image/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + str(step).zfill(5) + ".jpg")
def draw_3d_point_all(inputs, outputs, labels, step): #
    #['Palm', 'Wrist1', 'Wrist2', 'Thumb.R1', 'Thumb.R2', 'Thumb.T', 'Index.R', 'Index.T', 'Mid.R', 'Mid.T', 'Ring.R', 'Ring.T', 'Pinky.R', 'Pinky.T', 'Mean']
    label_id = [0,0,1,1,2,2,3,3,4,4,4,5,5,5]

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
    plt.savefig(directory + str(step).zfill(5) + ".jpg")

def visualize(fetch_results):
    shape = np.shape(fetch_results['initial_node_features'])
    graph_num = int(shape[0]/node_num)
    print("绘制"+str(graph_num)+"张图片")
    for step in range(graph_num):
        # draw(fetch_results['initial_node_features'][step*node_num:(step+1)*node_num,:],
        #           fetch_results['target_values'][step*node_num:(step+1)*node_num,:],
        #           fetch_results['final_output_node_representations'][step*node_num:(step+1)*node_num,:],
        #           fetch_results['initial_node_features_select'], step)
        draw_3d_point_all(fetch_results['initial_node_features'][step*node_num:(step+1)*node_num,:],
                        fetch_results['final_output_node_representations'][step*node_num:(step+1)*node_num,:],
                        fetch_results['target_values'][step*node_num:(step+1)*node_num,:],
                          step)


def test(model_path: str, test_data_path: Optional[RichPath], result_dir: str, quiet: bool = False):
    model = restore(model_path, result_dir)
    test_data_path = test_data_path or RichPath.create(model.task.default_data_path())
    create_pb = True
    if create_pb:
        # model.sess = tf.Session(graph=self.graph, config=config)
        # with model.graph.as_default():

        input_graph_def = model.graph.as_graph_def()
        # variable_names = [v.name for v in input_graph_def.node]
        # print(variable_names)
        # for op in model.graph.get_operations():
        #     print(str(op.name))
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            model.sess,  # The session
            input_graph_def,  # input_graph_def is useful for retrieving the nodes
            'out_layer_task0/final_output_node_representations'.split(",")
        )
        with tf.gfile.FastGFile(model_path[:-6]+"pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())
        """
        source activate TFlite
        tflite_convert \
        --graph_def_file=/home/chen/Documents/Mobile_hand_tf-gnn-samples-master/data/hand_gen/pool/point/5tip_xy/trained_model/HAND_GEN_GGNN_2019-07-27-22-02-20_7901_best_model.pb \
        --output_file=/home/chen/Documents/Mobile_hand_tf-gnn-samples-master/data/hand_gen/pool/point/5tip_xy/trained_model/HAND_GEN_GGNN_2019-07-27-22-02-20_7901_best_model.lite \
        --output_format=TFLITE \
        --input_shapes=32,3 \
        --input_arrays=initial_node_features \
        --output_arrays=out_layer_task0/final_output_node_representations \
        --inference_type=FLOAT
        
        """
    model.test(test_data_path)
    visualize(model.fetch_results)


def run(args):
    model_path = save_dataset_dir + "trained_model/HAND_GEN_GGNN_2019-08-23-10-19-19_19476_best_model.pickle"
    test_data_path = save_dataset_dir
    if test_data_path is not None:
        test_data_path = RichPath.create(test_data_path)
    result_dir = args.get('--result_dir', 'trained_models')
    test(model_path, test_data_path, result_dir, quiet=args.get('--quiet'))


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), enable_debugging=args['--debug'])

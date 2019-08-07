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

test_model = 'rich/'
level_model = 'point/'
detal_name = 'infer_z/'
save_dataset_dir = "data/hand_gen/"+test_model+level_model+detal_name

def draw(input, target, result, select, step):
    #观察序列，查看关键点坐标，确定角度由哪些坐标计算
    fig = plt.figure(1)
    fig_color = ['c', 'm', 'y', 'g', 'r', 'b']
    fig.clear()

    for ax_id in range(3):
        if ax_id==0:
            ax = fig.add_subplot(221, projection='3d')
            draw = input
            title = "input"
        elif ax_id==1:
            ax = fig.add_subplot(222, projection='3d')
            draw = target
            title = "target"
        else:
            ax = fig.add_subplot(223, projection='3d')
            draw = result
            title = "result"
        ax.set_title(title)
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
                    ax.plot([draw[f * 6+bone, 0], draw[f * 6 +bone+ 1, 0]],
                            [draw[f * 6+bone, 1], draw[f * 6 +bone+ 1, 1]],
                            [draw[f * 6+bone, 2], draw[f * 6 +bone+ 1, 2]], color=fig_color[f], linewidth=2)
                if f == 4:
                    ax.plot([draw[f * 6 + bone + 1, 0], draw[30, 0]],
                            [draw[f * 6 + bone + 1, 1], draw[30, 1]],
                            [draw[f * 6 + bone + 1, 2], draw[30, 2]], color=fig_color[f], linewidth=2)
                else:
                    ax.plot([draw[f * 6 + bone + 1, 0], draw[31, 0]],
                            [draw[f * 6 + bone + 1, 1], draw[31, 1]],
                            [draw[f * 6 + bone + 1, 2], draw[31, 2]], color=fig_color[f], linewidth=2)
            # ax.scatter(uvd_pt[f * 2, 0], uvd_pt[f * 2, 1], uvd_pt[f * 2, 2], s=60, c=fig_color[f])
            ax.scatter(draw[f*6:(f+1)*6, 0], draw[f*6:(f+1)*6, 1], draw[f*6:(f+1)*6, 2], s=30, c=fig_color[f])

    # plt.show()
    # plt.pause(0.01)
    directory = save_dataset_dir+"trained_model/test/image/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + str(step).zfill(5) + ".jpg")

def visualize(fetch_results):
    shape = np.shape(fetch_results['initial_node_features'])
    graph_num = int(shape[0]/32)
    print("绘制"+str(graph_num)+"张图片")
    for step in range(graph_num):
        draw(fetch_results['initial_node_features'][step*32:(step+1)*32,:],
                  fetch_results['target_values'][step*32:(step+1)*32,:],
                  fetch_results['final_output_node_representations'][step*32:(step+1)*32,:],
                  fetch_results['initial_node_features_select'], step)


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
    model_path = save_dataset_dir + "trained_model/HAND_GEN_GGNN_2019-07-28-16-25-42_3568_best_model.pickle"
    test_data_path = save_dataset_dir
    if test_data_path is not None:
        test_data_path = RichPath.create(test_data_path)
    result_dir = args.get('--result_dir', 'trained_models')
    test(model_path, test_data_path, result_dir, quiet=args.get('--quiet'))


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), enable_debugging=args['--debug'])

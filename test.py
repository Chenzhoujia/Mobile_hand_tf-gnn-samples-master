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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

from utils.model_utils import restore
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
    model.test(test_data_path)
    visualize(model.fetch_results)


def run(args):
    azure_info_path = args.get('--azure-info', None)
    model_path = args['STORED_MODEL_PATH']
    test_data_path = args.get('DATA_PATH')
    if test_data_path is not None:
        test_data_path = RichPath.create(test_data_path, azure_info_path)
    result_dir = args.get('--result_dir', 'trained_models')
    test(model_path, test_data_path, result_dir, quiet=args.get('--quiet'))


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), enable_debugging=args['--debug'])

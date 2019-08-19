import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import pandas as pd
import numpy as np
import scipy.io as sio
import _pickle as cPickle
import time, os, math
import collections
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import math
#加载数据, 筛选重要的，三个相机都可以用
src_dir = '/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/NYU/train/'
test_model = 'rich/'
level_model = 'point/'
detal_name = 'Method_disturbance_fc/'
methon_name = 'CVPR18_NYU_denseReg'
detal_name += methon_name + '/'

save_dataset_dir = "data/hand_gen/"+test_model+level_model+detal_name

def load_Method_disturbance_fc():
    path = os.path.join(src_dir, 'joint_data.mat')
    mat = sio.loadmat(path)
    joint_uvd = mat['joint_uvd']
    joint_uvd = joint_uvd.reshape(-1,36,3)
    #选择
    select_list =[0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32]
    delete_list = list(range(36))
    for i in select_list:
        delete_list.remove(i)
    joint_uvd = np.delete(joint_uvd,delete_list,axis=1)
    #转换到世界坐标系 单位mm
    params = get_param('nyu')
    joint_uvd = pixel2world(joint_uvd, *params)
    joint_uvd = np.vstack((joint_uvd,joint_uvd))
    #打乱顺序
    permutation = np.random.permutation(joint_uvd.shape[0])
    joint_uvd = joint_uvd[permutation, :, :]
    joint_uvd = normlize(joint_uvd)

    data_shape = joint_uvd.shape

    for file_path_id, file_path in enumerate(test_file_list_clear):
        output = np.loadtxt(file_path)
        output = np.reshape(output, (-1, 14, 3))

        ground_truth = np.loadtxt(
            '/home/chen/Documents/awesome-hand-pose-estimation-master/evaluation/groundtruth/nyu/nyu_test_groundtruth_label.txt')
        ground_truth = np.reshape(ground_truth, (-1, 14, 3))

        params = get_param('nyu')
        ground_truth = pixel2world(ground_truth, *params)
        output = pixel2world(output, *params)

        ground_truth = normlize(ground_truth)
        output = normlize(output)

        if file_path_id==0:
            move_bias = output - ground_truth
        else:
            move_bias = np.vstack((output - ground_truth, move_bias))


    # 生成两个随机整数0~8251和一个随机分量0~1 data_shape[0] 个
    move_bias = np.tile(move_bias,(math.ceil(data_shape[0]/move_bias.shape[0]),1,1))

    permutation = np.random.permutation(move_bias.shape[0])
    move_bias = move_bias[permutation, :, :]
    move_bias = move_bias[:data_shape[0]]

    output = move_bias + joint_uvd
    # 增加异常
    # proportion_array = np.random.rand(data_shape[0])
    # proportion_array = np.expand_dims(proportion_array,axis=-1)
    # proportion_array = np.expand_dims(proportion_array,axis=-1)
    # proportion_array = np.tile(proportion_array,(1,data_shape[1],data_shape[2]))

    # permutation1 = np.random.permutation(move_bias.shape[0])
    # permutation2 = np.random.permutation(move_bias.shape[0])
    #
    # move_bias = move_bias[permutation1, :, :]*proportion_array \
    #             + move_bias[permutation2, :, :]*(1-proportion_array)

    return joint_uvd, output


def normlize(joint_uvd):
    joint_xyz_norm = []
    for i in range(3):
        joint_min = np.min(joint_uvd[:,:,i], axis=1)[:,np.newaxis]
        joint_max = np.max(joint_uvd[:,:,i], axis=1)[:,np.newaxis]
        joint_xyz_norm.append((joint_uvd[:,:,i]-joint_min)/(joint_max - joint_min))
    joint_xyz_norm = np.concatenate([joint_xyz_norm[0][:,:,np.newaxis],joint_xyz_norm[1][:,:,np.newaxis],joint_xyz_norm[2][:,:,np.newaxis]],axis = -1)
    joint_xyz_norm = (joint_xyz_norm-0.5)*2
    return joint_xyz_norm
def get_param(dataset):
    if dataset == 'icvl':
        return 240.99, 240.96, 160, 120
    elif dataset == 'nyu':
        return 588.03, -587.07, 320, 240
    elif dataset == 'msra':
        return 241.42, 241.42, 160, 120


def pixel2world(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x

def get_file_path(root_path,file_list):
    #获取该目录下所有的文件名称和目录名称
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        #获取目录或者文件的路径
        dir_file_path = os.path.join(root_path,dir_file)
        #判断该路径为文件还是路径
        if os.path.isdir(dir_file_path):
            #递归获取所有文件和目录的路径
            get_file_path(dir_file_path,file_list)
        else:
            file_list.append(dir_file_path)

def draw_3d_point(labels, outputs, step, method):
    #['Palm', 'Wrist1', 'Wrist2', 'Thumb.R1', 'Thumb.R2', 'Thumb.T', 'Index.R', 'Index.T', 'Mid.R', 'Mid.T', 'Ring.R', 'Ring.T', 'Pinky.R', 'Pinky.T', 'Mean']
    label_id = [0,0,1,1,2,2,3,3,4,4,4,5,5,5]

    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')

    bone_len = []
    bone_fake_len = []
    for draw_i in range(2):
        if draw_i == 0:
            pose_show = labels
            fig_color = ['b', 'b', 'b', 'b', 'b', 'b']
        else:
            pose_show = outputs
            fig_color = ['r', 'r', 'r', 'r', 'r', 'r']

        ax.view_init(azim=20.0, elev=40.0)  # aligns the 3d coord with the camera view
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
        ax.set_zlim((-1, 1))
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
    if not os.path.isdir("./data/image/"):
        os.makedirs("./data/image/")
    plt.savefig("./data/image/" + method + str(step).zfill(10) + ".png")

def save_obj(obj, name):
    if not os.path.isdir(save_dataset_dir):
        os.makedirs(save_dataset_dir)
    with open(save_dataset_dir+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(save_dataset_dir + name + '.pkl', 'rb') as f:
        return pickle.load(f)

method_name = 'hand_test_CVPR18_NYU_DeepPrior++_FM'
a = load_obj(method_name)
step = 0
for sample in tqdm(a):
    draw_3d_point(sample["targets"], sample["node_features"], step, method_name)
    step = step + 1

#开始写文件
test_file_list = []
test_file_list_clear = []
method_name = []
get_file_path('/home/chen/Documents/awesome-hand-pose-estimation-master/evaluation/results/nyu/', test_file_list)
for file_path in test_file_list:
    if file_path.endswith('.txt') and ('19' in file_path):
        file_path_list = file_path.split('/')
        file_path_list = file_path_list[-1][:-4]
        method_name.append(file_path_list)
        test_file_list_clear.append(file_path)


test_dict_all = []
for file_path_id, file_path in enumerate(test_file_list_clear):
    print(file_path)
    output = np.loadtxt(file_path)
    output = np.reshape(output, (-1, 14, 3))
    hand_dict_array = []

    ground_truth = np.loadtxt(
        '/home/chen/Documents/awesome-hand-pose-estimation-master/evaluation/groundtruth/nyu/nyu_test_groundtruth_label.txt')
    ground_truth = np.reshape(ground_truth, (-1, 14, 3))
    joint_uvd = ground_truth
    move_bias = output
    params = get_param('nyu')
    joint_uvd = pixel2world(joint_uvd, *params)
    move_bias = pixel2world(move_bias, *params)

    joint_uvd = normlize(joint_uvd)
    move_bias = normlize(move_bias)

    for id in tqdm(range(0,joint_uvd.shape[0])):
        one_joint_xyz_norm = joint_uvd[id, :, :]
        one_joint_xyz_norm_mask = move_bias[id, :, :]

        test_dict = {"targets": one_joint_xyz_norm,
                     "graph": [[0, 1, 1]],
                     "id": "handGen: " + str(id).zfill(7),
                     "node_features": one_joint_xyz_norm_mask}

        hand_dict_array.append(test_dict)
        test_dict_all.append(test_dict)
        #draw_3d_point(one_joint_xyz_norm, one_joint_xyz_norm_mask, id, method_name[file_path_id])
    save_obj(hand_dict_array, 'hand_test_' + method_name[file_path_id])
save_obj(test_dict_all, 'hand_test')


# 保存train
joint_uvd, move_bias = load_Method_disturbance_fc()

hand_dict_array = []

for id in tqdm(range(joint_uvd.shape[0])):
    one_joint_xyz_norm = joint_uvd[id,:,:]
    one_joint_xyz_norm_mask = move_bias[id,:,:]

    test_dict = {"targets": one_joint_xyz_norm,
    "graph": [[0, 1, 1]],
    "id": "handGen: "+str(id).zfill(7),
    "node_features": one_joint_xyz_norm_mask}

    hand_dict_array.append(test_dict)

save_obj(hand_dict_array,'hand_train')

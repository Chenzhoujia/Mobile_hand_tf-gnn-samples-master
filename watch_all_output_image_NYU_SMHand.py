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
src_dir_test = '/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/NYU/test/'
test_model = 'rich/'
level_model = 'point/'
detal_name = 'smart_motion_hand/'
methon_name = 'tip_control'
detal_name += methon_name + '/'
save_dataset_dir = "data/hand_gen/"+test_model+level_model+detal_name

is_normlize = True
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
def mask_control_point(joint_uvd, list_point):
    shape = np.shape(joint_uvd)
    joint_uvd_keep = np.zeros_like(joint_uvd)
    for i in range(shape[0]):
        for j in list_point:
            joint_uvd_keep[i,j,:] = joint_uvd[i,j,:]
    return joint_uvd_keep
def hand_move(joint_uvd, time_step = 1):
    joint_uvd_move = joint_uvd[time_step:]
    for i in range(time_step):
        joint_uvd_move = np.vstack((joint_uvd_move, joint_uvd_move[-(i+2)][np.newaxis, :, :]))

    return joint_uvd_move

def normlize(joint_uvd):
    joint_xyz_norm = []
    for i in range(3):
        joint_mean = np.mean(joint_uvd[:,:,i], axis=1)[:,np.newaxis]
        joint_mean = 0.0
        joint_xyz_norm.append((joint_uvd[:,:,i]-joint_mean)/220)
    joint_xyz_norm = np.concatenate([joint_xyz_norm[0][:,:,np.newaxis],joint_xyz_norm[1][:,:,np.newaxis],joint_xyz_norm[2][:,:,np.newaxis]],axis = -1)

    return joint_xyz_norm
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
        # if not method.startswith('no_norm_'):
        # ax.set_xlim((-1, 0))
        # ax.set_ylim((-1, 0))
        # ax.set_zlim((3, 4))
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
    if not os.path.isdir("./data/image/"):
        os.makedirs("./data/image/")
    plt.savefig("./data/image/" + method + str(step).zfill(10) + ".png")

def save_obj_method(obj, name, method):
    if not os.path.isdir(save_dataset_dir + method + '/'):
        os.makedirs(save_dataset_dir + method + '/')
    with open(save_dataset_dir + method + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def save_obj(obj, name):
    if not os.path.isdir(save_dataset_dir + '/'):
        os.makedirs(save_dataset_dir + '/')
    with open(save_dataset_dir + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj_method(name, method):
    with open(save_dataset_dir + method + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_obj(name):
    with open(save_dataset_dir + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# #如果文件夹不存在就创建，如果文件存在就清空！
import shutil
if False:
    tmp_image = "./data/image/"
    if not os.path.exists(tmp_image):
        os.mkdir(tmp_image)
    else:
        shutil.rmtree(tmp_image)
        os.mkdir(tmp_image)
    method_name = 'hand_test'
    a = load_obj(method_name)
    errors = []
    step = 0
    for sample_id in tqdm(range(len(a))):
        sample =a[sample_id + 77 * 5]
        draw_3d_point(sample["targets"], sample["node_features"], step, method_name)
        step = step + 1
        errors.append(np.mean(np.sqrt(np.sum((sample["targets"] - sample["node_features"]) ** 2, axis=1))))
    errors = np.mean(np.array(errors))


#开始写文件

# 遍历两个数组

# 保存测试集合
hand_dict_array = []

path = os.path.join(src_dir_test, 'joint_data.mat')
mat = sio.loadmat(path)
joint_uvd = mat['joint_uvd']
joint_uvd = joint_uvd.reshape(-1, 36, 3)

params = get_param('nyu')
joint_uvd = pixel2world(joint_uvd, *params)

if is_normlize:
    joint_uvd = normlize(joint_uvd)
joint_uvd_move = hand_move(joint_uvd)
joint_uvd_move_keep = mask_control_point(joint_uvd_move, [0,6,12,18,24])

for id in tqdm(range(joint_uvd.shape[0])):
    node_features = joint_uvd[id, :, :]
    targets = joint_uvd_move[id, :, :]
    targets_control = joint_uvd_move_keep[id, :, :]

    test_dict = {"targets": targets,
                 "targets_control": targets_control,
                 "graph": [[0, 1, 1]],
                 "id": "handGen: " + str(id).zfill(7),
                 "node_features": node_features}

    hand_dict_array.append(test_dict)

save_obj(hand_dict_array, 'hand_test')


# 保存训练集合

# 保存train
hand_dict_array = []

path = os.path.join(src_dir, 'joint_data.mat')
mat = sio.loadmat(path)
joint_uvd = mat['joint_uvd']
joint_uvd = joint_uvd.reshape(-1, 36, 3)

params = get_param('nyu')
joint_uvd = pixel2world(joint_uvd, *params)

if is_normlize:
    joint_uvd = normlize(joint_uvd)
joint_uvd_move = hand_move(joint_uvd)
joint_uvd_move_keep = mask_control_point(joint_uvd_move, [0,6,12,18,24])

for id in tqdm(range(joint_uvd.shape[0])):
    node_features = joint_uvd[id, :, :]
    targets = joint_uvd_move[id, :, :]
    targets_control = joint_uvd_move_keep[id, :, :]

    test_dict = {"targets": targets,
                 "targets_control": targets_control,
                 "graph": [[0, 1, 1]],
                 "id": "handGen: " + str(id).zfill(7),
                 "node_features": node_features}


    hand_dict_array.append(test_dict)

save_obj(hand_dict_array, 'hand_train')

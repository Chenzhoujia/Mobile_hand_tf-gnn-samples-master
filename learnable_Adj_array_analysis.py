import shutil

import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
test_model = 'rich/'
level_model = 'point/'
detal_name = 'smart_motion_hand/'
methon_name = 'None_control'
detal_name += methon_name + '/'
save_dataset_dir = "data/hand_gen/"+test_model+level_model+detal_name
save_dataset_dir += "trained_model/"


tmp_image = save_dataset_dir+"learnable_ADJ/"
if not os.path.exists(tmp_image):
    os.mkdir(tmp_image)
else:
    shutil.rmtree(tmp_image)
    os.mkdir(tmp_image)

learnable_Adj_array = np.load(save_dataset_dir+"learnable_Adj_array.npy")
preloss_array = np.load(save_dataset_dir+"preloss_List.npy")
time_message_aggregation_array = np.load(save_dataset_dir+"time_message_aggregation_array.npy")
shape = np.shape(time_message_aggregation_array)
plt.clf()
plt.pcolor(time_message_aggregation_array)
plt.colorbar()
plt.savefig(save_dataset_dir+"learnable_ADJ/" + "time_message_aggregation_array.jpg")

shape = np.shape(learnable_Adj_array)

# 绘制loss变化
plt.clf()
plt.pcolor(preloss_array[-50:,:])
plt.colorbar()
plt.savefig(save_dataset_dir +"learnable_ADJ/final_loss.jpg")
#绘制矩阵变化
for step in range(int(shape[0]/10)):
    plt.clf()
    plt.pcolor(learnable_Adj_array[step*10], vmin=-0.5, vmax=0.5)
    plt.colorbar()
    plt.savefig(save_dataset_dir+"learnable_ADJ/"+ str(step).zfill(5) + ".jpg")

plt.clf()
plt.pcolor(learnable_Adj_array[shape[0]-1], vmin=-0.5, vmax=0.5)
plt.colorbar()
plt.savefig(save_dataset_dir+"learnable_ADJ/" + "final.jpg")

#统计行列
abs = np.abs(learnable_Adj_array[shape[0]-1])
abs_hang = abs.sum(axis=0)
abs_lie = abs.sum(axis=1)

plt.clf()
plt.pcolor(abs, vmin=0.0, vmax=0.5)
plt.colorbar()
plt.savefig(save_dataset_dir+"learnable_ADJ/" + "final_abs.jpg")

plt.clf()
plt.pcolor(np.expand_dims(abs_hang, 0))
plt.colorbar()
plt.savefig(save_dataset_dir+"learnable_ADJ/" + "final_sum0.jpg")

plt.clf()
plt.pcolor(np.expand_dims(abs_lie, 1))
plt.colorbar()
plt.savefig(save_dataset_dir+"learnable_ADJ/" + "final_sum1.jpg")

#统计手指之间的耦合
finger_relate = np.zeros([5,5])
for i in range(5):
    for j in range(5):
        finger_relate[i,j] = np.sum(abs[i*6:(i+1)*6,j*6:(j+1)*6])
plt.clf()
plt.pcolor(finger_relate)
plt.colorbar()
plt.savefig(save_dataset_dir+"learnable_ADJ/" + "final_finger_relate.jpg")
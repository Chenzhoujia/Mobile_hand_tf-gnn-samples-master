# -*- coding: utf-8 -*-
"""
Created on Sat May 19 17:34:54 2018

@author: xxx
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

def nothing(x):
    pass

# 当鼠标按下时变为 True
drawing = False
# 如果 mode 为 True 绘制矩形。按下 'm' 变成绘制曲线
mode = 'x'
ix, iy = -1, -1
ixl, iyl, izl = [],[],[]
#创建回调函数
def draw_circle(event, x, y, flags, param):
    r = cv.getTrackbarPos('R', 'image')
    g = cv.getTrackbarPos('G', 'image')
    b = cv.getTrackbarPos('B', 'image')
    color = (b, g, r)

    global ix, iy, drawing, mode, ixl, iyl   # 在函数内无法直接使用全局变量。
    # 当按下左键是返回起始位置坐标
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
#        当鼠标左键按下并移动是绘制图形。event 可以查看移动, flag 查看是否按下
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        if drawing == True:
            if mode == 'x':
                # 绘制圆圈，小圆点连在一起就成了线，3代表画笔的粗细
                #cv.circle(img, (ix, iy), 3, color, -1)
                cv.circle(img, (x, y), 3, color, -1)
                ixl.append(x)
                iyl.append(y)
            elif mode == 'z':
                cv.circle(img, (x, y), 3, (0, 255, 0), -1)
                izl.append(x)

#        当鼠标松开停止绘画
    elif event == cv.EVENT_LBUTTONUP:
            drawing == False

#创建一幅黑色图形
img = np.zeros((512, 512, 3), np.uint8)
cv.namedWindow('image')

cv.createTrackbar('R', 'image', 255, 255, nothing)
cv.createTrackbar('G', 'image', 0, 255, nothing)
cv.createTrackbar('B', 'image', 0, 255, nothing)
cv.setMouseCallback('image', draw_circle)

while(1):
    cv.imshow('image', img)
    k = cv.waitKey(1)&0xFF
    if k == ord('x'):
        mode = 'x'
    elif k == ord('z'):
        mode = 'z'
    elif k==27:
        break


cv.destroyAllWindows()

# 基本的数据处理

min_len = min(len(ixl),len(izl))
ixl = np.array(ixl[:min_len])
iyl = np.array(iyl[:min_len])
izl = np.array(izl[:min_len])

ixl = (ixl -ixl[0])/511
iyl = (iyl -iyl[0])/511
izl = (izl -izl[0])/511

fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(azim=20.0, elev=40.0)  # aligns the 3d coord with the camera view
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim((-1, 1))
ax.set_ylim((-1, 1))
ax.set_zlim((-1, 1))

ax.scatter(ixl, iyl, izl, s=10, c='r')
ax.scatter(ixl[-1], iyl[-1], izl[-1], s=20, c='b')
plt.show()
plt.pause(0.01)

data = np.stack([ixl,iyl,izl])

test_model = 'rich/'
level_model = 'point/'
detal_name = 'smart_motion_hand/'
methon_name = 'tip_control'
detal_name += methon_name + '/'
save_dataset_dir = "data/hand_gen/"+test_model+level_model+detal_name

np.save(save_dataset_dir+"test"+time.strftime('%Y-%m-%d %H:%M:%S')+".npy",data)
print(np.load(save_dataset_dir+"test"+time.strftime('%Y-%m-%d %H:%M:%S')+".npy"))
plt.savefig(save_dataset_dir+"test"+time.strftime('%Y-%m-%d %H:%M:%S')+".png")
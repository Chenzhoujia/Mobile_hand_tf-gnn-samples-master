import numpy as np
import matplotlib.pyplot as plt
# hc_array = np.zeros([32,32])
# for hc_i in range(32):
#     hc_array[hc_i,hc_i] = 1
#     if hc_i!=31 and (hc_i+1)%6!=0:
#         hc_array[hc_i,hc_i+1] = 1
#         hc_array[hc_i + 1, hc_i] = 1
# for hc_i in range(5):
#     hc_array[hc_i * 6 + 5, 30] = 1
#     hc_array[30, hc_i * 6 + 5] = 1
# hc_array[29, 31] = 1
# hc_array[31, 29] = 1

hc_array = np.zeros([32,32])
for hc_i in range(32):
    hc_array[hc_i,hc_i] = 1
    if hc_i!=31 and (hc_i+1)%6!=0:
        hc_array[hc_i,hc_i+1] = 1
        hc_array[hc_i + 1, hc_i] = 1
for hc_i in range(5):
    hc_array[hc_i * 6 + 5, 30] = 1
    hc_array[30, hc_i * 6 + 5] = 1
hc_array[29, 31] = 1
hc_array[31, 29] = 1

for hc_i2 in range(24):
    hc_array[hc_i2, hc_i2+6] =  1
    hc_array[hc_i2+6, hc_i2] =  1


plt.clf()
plt.pcolor(hc_array)
plt.colorbar()
plt.show()
plt.pause(0.01)
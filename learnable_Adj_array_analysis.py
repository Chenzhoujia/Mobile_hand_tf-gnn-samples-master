import numpy as np

learnable_Adj_array = np.load("./learnable_Adj_array.npy")
preloss_array = np.load("./preloss_List.npy")
shape = np.shape(learnable_Adj_array)
import numpy as np
import matplotlib.pyplot as plt

for step in range(shape[0]):
    plt.clf()
    plt.pcolor(learnable_Adj_array[step], vmin=-0.5, vmax=0.5)
    plt.colorbar()

    plt.savefig("./learnable_ADJ/" + str(step).zfill(5) + ".jpg")

plt.clf()
plt.pcolor(preloss_array)

plt.colorbar()

plt.savefig("./learnable_ADJ/" + str(0).zfill(5) + "_loss.jpg")
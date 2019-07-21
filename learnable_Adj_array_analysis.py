import numpy as np

learnable_Adj_array = np.load("./learnable_Adj_array.npy")
shape = np.shape(learnable_Adj_array)
import numpy as np
import matplotlib.pyplot as plt

for step in range(shape[0]):
    plt.clf()
    plt.pcolor(learnable_Adj_array[step], vmin=-0.5, vmax=0.5)
    #plt.imshow(learnable_Adj_array[step])
    plt.colorbar()

    plt.savefig("./learnable_ADJ/" + str(step).zfill(5) + ".jpg")

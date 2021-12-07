import numpy as np
import imageio as im
import matplotlib.pyplot as plt
import func_Photomontage as ph

I = np.array([np.array(im.imread("data\image_0"+str(i)+".png")).astype(int) for i in range(1,6)]).astype(int)
M = (np.array([np.array(im.imread("data\mask_0"+str(i)+".png")).astype(int) for i in range(1,6)]) / 255).astype(int)

plt.imsave('result.png', ph.Photomontage(I, M))
print('Done')
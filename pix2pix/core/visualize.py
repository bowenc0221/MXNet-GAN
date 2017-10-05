# Visualize training
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Plots two images side by side using matplotlib
def visualize(real_A, real_B, fake_B, fname):
    # 1x3x256x256 to 1x256x256x3
    real_A = real_A.transpose((0, 2, 3, 1))
    real_B = real_B.transpose((0, 2, 3, 1))
    fake_B = fake_B.transpose((0, 2, 3, 1))
    # Pixel values from 0-255
    real_A = np.clip((real_A + 1.0) * (255.0 / 2.0), 0, 255).astype(np.uint8)
    real_B = np.clip((real_B + 1.0) * (255.0 / 2.0), 0, 255).astype(np.uint8)
    fake_B = np.clip((fake_B + 1.0) * (255.0 / 2.0), 0, 255).astype(np.uint8)

    # Create a matplotlib figure with two subplots: one for the real and the other for the fake
    # fill each plot with our buffer array, which creates the image
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_title('Input')
    ax1.axis('off')
    ax1.imshow(real_A[0])
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.imshow(real_B[0])
    ax2.set_title('Ground Truth')
    ax2.axis('off')
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.imshow(fake_B[0])
    ax3.set_title('pix2pix')
    ax3.axis('off')
    # plt.show()
    plt.savefig(fname)
    plt.close()
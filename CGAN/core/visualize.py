# Visualize training
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


# Takes the images in our batch and arranges them in an array so that they can be
# Plotted using matplotlib
def fill_buf(buf, num_images, img, shape):
    width = buf.shape[0] / shape[1]
    height = buf.shape[1] / shape[0]
    img_width = (num_images % width) * shape[0]
    img_hight = (num_images / height) * shape[1]
    buf[img_hight:img_hight + shape[1], img_width:img_width + shape[0], :] = img


# Plots two images side by side using matplotlib
def visualize(fake, real, fname):
    # 64x3x64x64 to 64x64x64x3
    fake = fake.transpose((0, 2, 3, 1))
    # Pixel values from 0-255
    fake = np.clip((fake + 1.0) * (255.0 / 2.0), 0, 255).astype(np.uint8)
    # Repeat for real image
    real = real.transpose((0, 2, 3, 1))
    real = np.clip((real + 1.0) * (255.0 / 2.0), 0, 255).astype(np.uint8)

    # Create buffer array that will hold all the images in our batch
    # Fill the buffer so to arrange all images in the batch onto the buffer array
    n = np.ceil(np.sqrt(fake.shape[0]))
    fbuff = np.zeros((int(n * fake.shape[1]), int(n * fake.shape[2]), int(fake.shape[3])), dtype=np.uint8)
    for i, img in enumerate(fake):
        fill_buf(fbuff, i, img, fake.shape[1:3])
    rbuff = np.zeros((int(n * real.shape[1]), int(n * real.shape[2]), int(real.shape[3])), dtype=np.uint8)
    for i, img in enumerate(real):
        fill_buf(rbuff, i, img, real.shape[1:3])

    # Create a matplotlib figure with two subplots: one for the real and the other for the fake
    # fill each plot with our buffer array, which creates the image
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.axis('off')
    ax1.imshow(fbuff)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(rbuff)
    ax2.axis('off')
    # plt.show()
    plt.savefig(fname)
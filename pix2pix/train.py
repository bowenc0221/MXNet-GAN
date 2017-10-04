import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
this_dir = os.path.dirname(__file__)

from function import train_pix2pix
from function import test_pix2pix

if __name__ == "__main__":
    train_pix2pix.main()
    test_pix2pix.main()
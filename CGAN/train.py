import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
this_dir = os.path.dirname(__file__)

from function import train_dcgan
from function import test_dcgan

if __name__ == "__main__":
    train_dcgan.main()
    test_dcgan.main()
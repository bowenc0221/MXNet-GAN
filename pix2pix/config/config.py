# --------------------------------------------------------
# Generative Adversarial Net
# Copyright (c) 2016 by Contributors
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Bowen Cheng
# --------------------------------------------------------

import yaml
import numpy as np
from easydict import EasyDict as edict

config = edict()

config.MXNET_VERSION = ''
config.output_path = ''
config.symbol = ''
config.gpus = ''
config.RNG_SEED = 1

config.loadSize = 286
config.fineSize = 256
config.AtoB = False
config.netG = 'autoencoder'  # 'autoencoder' or 'unet'
config.netD = 'basic'  # 'basic' or 'n_layers'
config.n_layers = 0  # only used if netD=='n_layers'
config.GAN_loss = 1  # use GAN loss set to 1, do not use GAN loss set to 0

# default training
config.default = edict()
config.default.frequent = 20
config.default.kvstore = 'device'

# dataset related params
config.dataset = edict()
config.dataset.dataset = 'facades'
config.dataset.root = './data'
config.dataset.imageset = 'train'
config.dataset.image_root = './datasets'
config.dataset.testset = 'val'

config.TRAIN = edict()

config.TRAIN.optimizer = 'adam'
config.TRAIN.lr = 0.0002
config.TRAIN.beta1 = 0.5
config.TRAIN.beta2 = 0.999
config.TRAIN.momentum = 0.9
config.TRAIN.wd = 0.0005
config.TRAIN.begin_epoch = 0
config.TRAIN.end_epoch = 200
config.TRAIN.model_prefix = ''

config.TRAIN.step_epoch = 100
config.TRAIN.decay_epoch = 100

# whether resume training
config.TRAIN.RESUME = False
# whether shuffle image
config.TRAIN.SHUFFLE = True
config.TRAIN.FLIP = True
# batch size
config.TRAIN.BATCH_SIZE = 1

config.TRAIN.epoch_end_plot_figure = True
config.TRAIN.batch_end_plot_figure = False
config.TRAIN.save_interval = 20

# L1 loss weight
config.TRAIN.lambda_l1 = 100

config.TEST = edict()

config.TEST.TEST_EPOCH = 0
config.TEST.img_h = 256
config.TEST.img_w = 256


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    config[k] = v
            else:
                raise ValueError("key must exist in config.py")

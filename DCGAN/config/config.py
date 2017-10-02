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

# default training
config.default = edict()
config.default.frequent = 20
config.default.kvstore = 'device'

# dataset related params
config.dataset = edict()
config.dataset.dataset = 'mnist'

config.TRAIN = edict()

config.TRAIN.optimizer = 'adam'
config.TRAIN.lr = 0.001
config.TRAIN.beta1 = 0.9
config.TRAIN.beta2 = 0.999
config.TRAIN.lr_step = ''
config.TRAIN.lr_factor = 0.1
config.TRAIN.momentum = 0.9
config.TRAIN.wd = 0.0005
config.TRAIN.begin_epoch = 0
config.TRAIN.end_epoch = 0
config.TRAIN.model_prefix = ''

# whether resume training
config.TRAIN.RESUME = False
# whether shuffle image
config.TRAIN.SHUFFLE = True
# batch size
config.TRAIN.BATCH_SIZE = 64


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
                    if k == 'SCALES':
                        config[k] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("key must exist in config.py")

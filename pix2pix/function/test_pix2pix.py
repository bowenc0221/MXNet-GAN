import cv2
import time
import argparse
import logging
import pprint
import os
import sys
import matplotlib
matplotlib.use('Agg')
from config.config import config, update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Test PIX2PIX')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    args = parser.parse_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../../external/mxnet', config.MXNET_VERSION))

import shutil
import numpy as np
import mxnet as mx

from symbols.pix2pix import get_symbol_generator, get_symbol_generator_instance_autoencoder, get_symbol_generator_instance_unet, get_symbol_discriminator
from core.create_logger import create_logger
from core.loader import pix2pixIter
from core.visualize import visualize

def main():
    # =============setting============
    dataset = config.dataset.dataset
    batch_size = config.TRAIN.BATCH_SIZE
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    assert len(ctx) == 1
    ctx = ctx[0]
    epoch = config.TEST.TEST_EPOCH

    logger, final_output_path = create_logger(config.output_path, args.cfg)
    prefix = os.path.join(final_output_path, config.TRAIN.model_prefix)
    test_fig_path = os.path.join(final_output_path, 'test_fig')

    if not os.path.exists(test_fig_path):
        os.makedirs(test_fig_path)

    test_fig_prefix = os.path.join(test_fig_path, dataset)

    # mx.random.seed(config.RNG_SEED)
    # np.random.seed(config.RNG_SEED)

    # ==============data==============
    test_data = pix2pixIter(config, shuffle=False, ctx=ctx, is_train=False)

    # print config
    pprint.pprint(config)
    print 'mxnet path:{}'.format(mx.__file__)
    # logger.info('system:{}'.format(os.uname()))
    # logger.info('mxnet path:{}'.format(mx.__file__))
    # logger.info('rng seed:{}'.format(config.RNG_SEED))
    # logger.info('training config:{}\n'.format(pprint.pformat(config)))

    # =============Generator Module=============
    if batch_size == 1:
        if config.network == 'autoencoder':
            generatorSymbol = get_symbol_generator_instance_autoencoder(config)
        elif config.network == 'unet':
            generatorSymbol = get_symbol_generator_instance_unet(config)
        else:
            raise NotImplemented
    else:
        generatorSymbol = get_symbol_generator()
    generator = mx.mod.Module(symbol=generatorSymbol, data_names=('A', 'B',), label_names=None, context=ctx)
    generator.bind(data_shapes=test_data.provide_data)
    generator.load_params(prefix + '-generator-%04d.params' % epoch)

    test_data.reset()
    batch = test_data.next()
    generator.forward(batch, is_train=False)
    outG = generator.get_outputs()
    visualize(batch.data[0].asnumpy(), batch.data[1].asnumpy(), outG[0].asnumpy(), test_fig_prefix + '-test-%04d.png' % epoch)

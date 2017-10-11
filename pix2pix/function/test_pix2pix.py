import cv2
import time
import argparse
import logging
import pprint
import os
import sys
import matplotlib
matplotlib.use('Agg')
from skimage import io
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

from symbols.pix2pix_instance import defineG_encoder_decoder, defineG_unet, defineD_n_layers, defineD_basic
from symbols.pix2pix_batch import defineG_encoder_decoder_batch, defineG_unet_batch, defineD_n_layers_batch, defineD_basic_batch
from core.create_logger import create_logger
from core.loader import pix2pixIter

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
        if config.netG == 'autoencoder':
            generatorSymbol = defineG_encoder_decoder(config)
        elif config.netG == 'unet':
            generatorSymbol = defineG_unet(config)
        else:
            raise NotImplemented
    else:
        if config.netG == 'autoencoder':
            generatorSymbol = defineG_encoder_decoder_batch(config, is_train=False)
        elif config.netG == 'unet':
            generatorSymbol = defineG_unet_batch(config, is_train=False)
        else:
            raise NotImplemented

    generator = mx.mod.Module(symbol=generatorSymbol, data_names=('A', 'B',), label_names=None, context=ctx)
    generator.bind(data_shapes=test_data.provide_data)
    generator.load_params(prefix + '-generator-%04d.params' % epoch)

    # if batch_size > 1:
    #     # use test set statistic by setting mean and variance to zero
    #     aux_names = generatorSymbol.list_auxiliary_states()
    #     arg_params, aux_params = generator.get_params()
    #     for aux_name in aux_names:
    #         if 'mean' in aux_name:
    #             aux_params[aux_name] = mx.nd.zeros_like(aux_params[aux_name])
    #         elif 'var' in aux_name:
    #             aux_params[aux_name] = mx.nd.ones_like(aux_params[aux_name])
    #         else:
    #             raise NameError('Unknown aux_name.')
    #     generator.set_params(arg_params=arg_params, aux_params=aux_params)

    test_data.reset()
    count = 1
    for batch in test_data:
        generator.forward(batch, is_train=False)
        outG = generator.get_outputs()[1].asnumpy()
        fake_B = outG.transpose((0, 2, 3, 1))
        fake_B = np.clip((fake_B + 1.0) * (255.0 / 2.0), 0, 255).astype(np.uint8)
        for n in range(batch_size):
            fname = test_fig_prefix + '-test-%04d-%06d.png' % (epoch, count + n)
            io.imsave(fname, fake_B[n])
        count += batch_size

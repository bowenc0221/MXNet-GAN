# import cv2
# import time
# import argparse
# import logging
# import pprint
# import os
# import sys
# import matplotlib
# matplotlib.use('Agg')
# from config.config import config, update_config
# from symbols.mxnet_dcgan import get_symbol_generator, get_symbol_discriminator
# from core.dataset import get_mnist
# from core.create_logger import create_logger
# from core.loader import RandIter
# from core.visualize import visualize
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='Test DCGAN')
#     # general
#     parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
#
#     args, rest = parser.parse_known_args()
#     # update config
#     update_config(args.cfg)
#
#     args = parser.parse_args()
#     return args
#
# args = parse_args()
# curr_path = os.path.abspath(os.path.dirname(__file__))
# sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))
#
# import shutil
# import numpy as np
# import mxnet as mx
#
#
# def main():
#     # =============setting============
#     dataset = config.dataset.dataset
#     batch_size = config.TRAIN.BATCH_SIZE
#     Z = 100
#     ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
#     assert len(ctx) == 1
#     ctx = ctx[0]
#     epoch = config.TEST.TEST_EPOCH
#
#     logger, final_output_path = create_logger(config.output_path, args.cfg)
#     prefix = os.path.join(final_output_path, config.TRAIN.model_prefix)
#     test_fig_path = os.path.join(final_output_path, 'test_fig')
#
#     if not os.path.exists(test_fig_path):
#         os.makedirs(test_fig_path)
#
#     test_fig_prefix = os.path.join(test_fig_path, dataset)
#
#     mx.random.seed(config.RNG_SEED)
#     np.random.seed(config.RNG_SEED)
#
#     # ==============data==============
#     if dataset == 'mnist':
#         X_train, X_test = get_mnist()
#         test_iter = mx.io.NDArrayIter(X_test, batch_size=batch_size)
#     else:
#         raise NotImplemented
#
#     rand_iter = RandIter(batch_size, Z)
#
#     # print config
#     pprint.pprint(config)
#     # logger.info('system:{}'.format(os.uname()))
#     # logger.info('mxnet path:{}'.format(mx.__file__))
#     # logger.info('rng seed:{}'.format(config.RNG_SEED))
#     # logger.info('training config:{}\n'.format(pprint.pformat(config)))
#
#     # =============Generator Module=============
#     generatorSymbol = get_symbol_generator()
#     generator = mx.mod.Module(symbol=generatorSymbol, data_names=('rand',), label_names=None, context=ctx)
#     generator.bind(data_shapes=rand_iter.provide_data)
#     generator.load_params(prefix + '-generator-%04d.params' % epoch)
#
#     test_iter.reset()
#     batch = test_iter.next()
#     rbatch = rand_iter.next()
#     generator.forward(rbatch, is_train=False)
#     outG = generator.get_outputs()
#     visualize(outG[0].asnumpy(), batch.data[0].asnumpy(), test_fig_prefix + '-test-%04d.png' % epoch)

# Symbols for pix2pix
# Generator: learn A -> B
# Discriminator: (A, B) -> real/fake

import mxnet as mx

def defineG_encoder_decoder_batch(cfg):
    ngf = 64
    eps = 1e-5 + 1e-12

    use_global_stats = True

    real_A = mx.sym.Variable(name='A')
    real_B = mx.sym.Variable(name='B')

    # --- e1 ---- input is (nc) x 256 x 256
    down_conv1 = mx.sym.Convolution(data=real_A, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf,
                                    name='down_conv1')

    # --- e2 ---- input is (ngf) x 128 x 128
    down_relu2 = mx.sym.LeakyReLU(data=down_conv1, act_type='leaky', slope=0.2, name='down_relu2')
    down_conv2 = mx.sym.Convolution(data=down_relu2, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 2,
                                    no_bias=True, name='down_conv2')
    down_norm2 = mx.sym.BatchNorm(data=down_conv2, eps=eps, use_global_stats=use_global_stats, name='down_norm2')

    # --- e3 ---- input is (ngf * 2) x 64 x 64
    down_relu3 = mx.sym.LeakyReLU(data=down_norm2, act_type='leaky', slope=0.2, name='down_relu3')
    down_conv3 = mx.sym.Convolution(data=down_relu3, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 4,
                                    no_bias=True, name='down_conv3')
    down_norm3 = mx.sym.BatchNorm(data=down_conv3, eps=eps, use_global_stats=use_global_stats, name='down_norm3')

    # --- e4 ---- input is (ngf * 4) x 32 x 32
    down_relu4 = mx.sym.LeakyReLU(data=down_norm3, act_type='leaky', slope=0.2, name='down_relu4')
    down_conv4 = mx.sym.Convolution(data=down_relu4, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='down_conv4')
    down_norm4 = mx.sym.BatchNorm(data=down_conv4, eps=eps, use_global_stats=use_global_stats, name='down_norm4')

    # --- e5 ---- input is (ngf * 8) x 16 x 16
    down_relu5 = mx.sym.LeakyReLU(data=down_norm4, act_type='leaky', slope=0.2, name='down_relu5')
    down_conv5 = mx.sym.Convolution(data=down_relu5, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='down_conv5')
    down_norm5 = mx.sym.BatchNorm(data=down_conv5, eps=eps, use_global_stats=use_global_stats, name='down_norm5')

    # --- e6 ---- input is (ngf * 8) x 8 x 8
    down_relu6 = mx.sym.LeakyReLU(data=down_norm5, act_type='leaky', slope=0.2, name='down_relu6')
    down_conv6 = mx.sym.Convolution(data=down_relu6, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='down_conv6')
    down_norm6 = mx.sym.BatchNorm(data=down_conv6, eps=eps, use_global_stats=use_global_stats, name='down_norm6')

    # --- e7 ---- input is (ngf * 8) x 4 x 4
    down_relu7 = mx.sym.LeakyReLU(data=down_norm6, act_type='leaky', slope=0.2, name='down_relu7')
    down_conv7 = mx.sym.Convolution(data=down_relu7, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='down_conv7')
    down_norm7 = mx.sym.BatchNorm(data=down_conv7, eps=eps, use_global_stats=use_global_stats, name='down_norm7')

    # --- e8 ---- input is (ngf * 8) x 2 x 2
    down_relu8 = mx.sym.LeakyReLU(data=down_norm7, act_type='leaky', slope=0.2, name='down_relu8')
    down_conv8 = mx.sym.Convolution(data=down_relu8, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='down_conv8')

    # --- d1 ---- input is (ngf * 8) x 1 x 1
    up_relu8 = mx.sym.Activation(data=down_conv8, act_type='relu', name='up_relu8')
    up_conv8 = mx.sym.Deconvolution(data=up_relu8, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='up_conv8')
    up_norm8 = mx.sym.BatchNorm(data=up_conv8, eps=eps, use_global_stats=use_global_stats, name='up_norm8')
    up_drop8 = mx.sym.Dropout(data=up_norm8, p=0.5, mode='always', name='up_drop8')

    # --- d2 ---- input is (ngf * 8) x 2 x 2
    up_relu7 = mx.sym.Activation(data=up_drop8, act_type='relu', name='up_relu7')
    up_conv7 = mx.sym.Deconvolution(data=up_relu7, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='up_conv7')
    up_norm7 = mx.sym.BatchNorm(data=up_conv7, eps=eps, use_global_stats=use_global_stats, name='up_norm7')
    up_drop7 = mx.sym.Dropout(data=up_norm7, p=0.5, mode='always', name='up_drop7')

    # --- d3 ---- input is (ngf * 8) x 4 x 4
    up_relu6 = mx.sym.Activation(data=up_drop7, act_type='relu', name='up_relu6')
    up_conv6 = mx.sym.Deconvolution(data=up_relu6, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='up_conv6')
    up_norm6 = mx.sym.BatchNorm(data=up_conv6, eps=eps, use_global_stats=use_global_stats, name='up_norm6')
    up_drop6 = mx.sym.Dropout(data=up_norm6, p=0.5, mode='always', name='up_drop6')

    # --- d4 ---- input is (ngf * 8) x 8 x 8
    up_relu5 = mx.sym.Activation(data=up_drop6, act_type='relu', name='up_relu5')
    up_conv5 = mx.sym.Deconvolution(data=up_relu5, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='up_conv5')
    up_norm5 = mx.sym.BatchNorm(data=up_conv5, eps=eps, use_global_stats=use_global_stats, name='up_norm5')

    # --- d5 ---- input is (ngf * 8) x 16 x 16
    up_relu4 = mx.sym.Activation(data=up_norm5, act_type='relu', name='up_relu4')
    up_conv4 = mx.sym.Deconvolution(data=up_relu4, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 4,
                                    no_bias=True, name='up_conv4')
    up_norm4 = mx.sym.BatchNorm(data=up_conv4, eps=eps, use_global_stats=use_global_stats, name='up_norm4')

    # --- d6 ---- input is (ngf * 4) x 32 x 32
    up_relu3 = mx.sym.Activation(data=up_norm4, act_type='relu', name='up_relu3')
    up_conv3 = mx.sym.Deconvolution(data=up_relu3, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 2,
                                    no_bias=True, name='up_conv3')
    up_norm3 = mx.sym.BatchNorm(data=up_conv3, eps=eps, use_global_stats=use_global_stats, name='up_norm3')

    # --- d7 ---- input is (ngf * 2) x 64 x 64
    up_relu2 = mx.sym.Activation(data=up_norm3, act_type='relu', name='up_relu2')
    up_conv2 = mx.sym.Deconvolution(data=up_relu2, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf,
                                    no_bias=True, name='up_conv2')
    up_norm2 = mx.sym.BatchNorm(data=up_conv2, eps=eps, use_global_stats=use_global_stats, name='up_norm2')

    # --- d8 ---- input is (ngf) x128 x 128
    up_relu1 = mx.sym.Activation(data=up_norm2, act_type='relu', name='up_relu1')
    up_conv1 = mx.sym.Deconvolution(data=up_relu1, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=3,
                                    name='up_conv1')

    up_tanh = mx.sym.Activation(up_conv1, name='up_tanh', act_type='tanh')

    l1_loss_ = mx.sym.mean(mx.sym.abs(up_tanh - real_B), axis=(1, 2, 3))
    l1_loss = mx.sym.MakeLoss(l1_loss_, grad_scale=cfg.TRAIN.lambda_l1)

    group = mx.sym.Group([l1_loss, up_tanh])

    return group

def defineG_unet_batch(cfg):
    ngf = 64
    eps = 1e-5 + 1e-12

    use_global_stats = True

    real_A = mx.sym.Variable(name='A')
    real_B = mx.sym.Variable(name='B')

    # --- e1 ---- input is (nc) x 256 x 256
    down_conv1 = mx.sym.Convolution(data=real_A, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf,
                                    name='down_conv1')

    # --- e2 ---- input is (ngf) x 128 x 128
    down_relu2 = mx.sym.LeakyReLU(data=down_conv1, act_type='leaky', slope=0.2, name='down_relu2')
    down_conv2 = mx.sym.Convolution(data=down_relu2, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 2,
                                    no_bias=True, name='down_conv2')
    down_norm2 = mx.sym.BatchNorm(data=down_conv2, eps=eps, use_global_stats=use_global_stats, name='down_norm2')

    # --- e3 ---- input is (ngf * 2) x 64 x 64
    down_relu3 = mx.sym.LeakyReLU(data=down_norm2, act_type='leaky', slope=0.2, name='down_relu3')
    down_conv3 = mx.sym.Convolution(data=down_relu3, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 4,
                                    no_bias=True, name='down_conv3')
    down_norm3 = mx.sym.BatchNorm(data=down_conv3, eps=eps, use_global_stats=use_global_stats, name='down_norm3')

    # --- e4 ---- input is (ngf * 4) x 32 x 32
    down_relu4 = mx.sym.LeakyReLU(data=down_norm3, act_type='leaky', slope=0.2, name='down_relu4')
    down_conv4 = mx.sym.Convolution(data=down_relu4, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='down_conv4')
    down_norm4 = mx.sym.BatchNorm(data=down_conv4, eps=eps, use_global_stats=use_global_stats, name='down_norm4')

    # --- e5 ---- input is (ngf * 8) x 16 x 16
    down_relu5 = mx.sym.LeakyReLU(data=down_norm4, act_type='leaky', slope=0.2, name='down_relu5')
    down_conv5 = mx.sym.Convolution(data=down_relu5, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='down_conv5')
    down_norm5 = mx.sym.BatchNorm(data=down_conv5, eps=eps, use_global_stats=use_global_stats, name='down_norm5')

    # --- e6 ---- input is (ngf * 8) x 8 x 8
    down_relu6 = mx.sym.LeakyReLU(data=down_norm5, act_type='leaky', slope=0.2, name='down_relu6')
    down_conv6 = mx.sym.Convolution(data=down_relu6, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='down_conv6')
    down_norm6 = mx.sym.BatchNorm(data=down_conv6, eps=eps, use_global_stats=use_global_stats, name='down_norm6')

    # --- e7 ---- input is (ngf * 8) x 4 x 4
    down_relu7 = mx.sym.LeakyReLU(data=down_norm6, act_type='leaky', slope=0.2, name='down_relu7')
    down_conv7 = mx.sym.Convolution(data=down_relu7, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='down_conv7')
    down_norm7 = mx.sym.BatchNorm(data=down_conv7, eps=eps, use_global_stats=use_global_stats, name='down_norm7')

    # --- e8 ---- input is (ngf * 8) x 2 x 2
    down_relu8 = mx.sym.LeakyReLU(data=down_norm7, act_type='leaky', slope=0.2, name='down_relu8')
    down_conv8 = mx.sym.Convolution(data=down_relu8, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='down_conv8')

    # --- d1 ---- input is (ngf * 8) x 1 x 1
    up_relu8 = mx.sym.Activation(data=down_conv8, act_type='relu', name='up_relu8')
    up_conv8 = mx.sym.Deconvolution(data=up_relu8, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='up_conv8')
    up_norm8 = mx.sym.BatchNorm(data=up_conv8, eps=eps, use_global_stats=use_global_stats, name='up_norm8')
    up_drop8 = mx.sym.Dropout(data=up_norm8, p=0.5, mode='always', name='up_drop8')

    d1 = mx.sym.concat(up_drop8, down_norm7, dim=1, name='d1')

    # --- d2 ---- input is (ngf * 8) x 2 x 2
    up_relu7 = mx.sym.Activation(data=d1, act_type='relu', name='up_relu7')
    up_conv7 = mx.sym.Deconvolution(data=up_relu7, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='up_conv7')
    up_norm7 = mx.sym.BatchNorm(data=up_conv7, eps=eps, use_global_stats=use_global_stats, name='up_norm7')
    up_drop7 = mx.sym.Dropout(data=up_norm7, p=0.5, mode='always', name='up_drop7')

    d2 = mx.sym.concat(up_drop7, down_norm6, dim=1, name='d2')

    # --- d3 ---- input is (ngf * 8) x 4 x 4
    up_relu6 = mx.sym.Activation(data=d2, act_type='relu', name='up_relu6')
    up_conv6 = mx.sym.Deconvolution(data=up_relu6, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='up_conv6')
    up_norm6 = mx.sym.BatchNorm(data=up_conv6, eps=eps, use_global_stats=use_global_stats, name='up_norm6')
    up_drop6 = mx.sym.Dropout(data=up_norm6, p=0.5, mode='always', name='up_drop6')

    d3 = mx.sym.concat(up_drop6, down_norm5, dim=1, name='d3')

    # --- d4 ---- input is (ngf * 8) x 8 x 8
    up_relu5 = mx.sym.Activation(data=d3, act_type='relu', name='up_relu5')
    up_conv5 = mx.sym.Deconvolution(data=up_relu5, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='up_conv5')
    up_norm5 = mx.sym.BatchNorm(data=up_conv5, eps=eps, use_global_stats=use_global_stats, name='up_norm5')

    d4 = mx.sym.concat(up_norm5, down_norm4, dim=1, name='d4')

    # --- d5 ---- input is (ngf * 8) x 16 x 16
    up_relu4 = mx.sym.Activation(data=d4, act_type='relu', name='up_relu4')
    up_conv4 = mx.sym.Deconvolution(data=up_relu4, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 4,
                                    no_bias=True, name='up_conv4')
    up_norm4 = mx.sym.BatchNorm(data=up_conv4, eps=eps, use_global_stats=use_global_stats, name='up_norm4')

    d5 = mx.sym.concat(up_norm4, down_norm3, dim=1, name='d5')

    # --- d6 ---- input is (ngf * 4) x 32 x 32
    up_relu3 = mx.sym.Activation(data=d5, act_type='relu', name='up_relu3')
    up_conv3 = mx.sym.Deconvolution(data=up_relu3, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 2,
                                    no_bias=True, name='up_conv3')
    up_norm3 = mx.sym.BatchNorm(data=up_conv3, eps=eps, use_global_stats=use_global_stats, name='up_norm3')

    d6 = mx.sym.concat(up_norm3, down_norm2, dim=1, name='d6')

    # --- d7 ---- input is (ngf * 2) x 64 x 64
    up_relu2 = mx.sym.Activation(data=d6, act_type='relu', name='up_relu2')
    up_conv2 = mx.sym.Deconvolution(data=up_relu2, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf,
                                    no_bias=True, name='up_conv2')
    up_norm2 = mx.sym.BatchNorm(data=up_conv2, eps=eps, use_global_stats=use_global_stats, name='up_norm2')

    d7 = mx.sym.concat(up_norm2, down_conv1, dim=1, name='d7')

    # --- d8 ---- input is (ngf) x128 x 128
    up_relu1 = mx.sym.Activation(data=d7, act_type='relu', name='up_relu1')
    up_conv1 = mx.sym.Deconvolution(data=up_relu1, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=3,
                                    name='up_conv1')

    up_tanh = mx.sym.Activation(up_conv1, name='up_tanh', act_type='tanh')

    l1_loss_ = mx.sym.mean(mx.sym.abs(up_tanh - real_B), axis=(1, 2, 3))
    l1_loss = mx.sym.MakeLoss(l1_loss_, grad_scale=cfg.TRAIN.lambda_l1)

    group = mx.sym.Group([l1_loss, up_tanh])

    return group

def defineD_basic_batch(batch_size):
    n_layers = 3
    return defineD_n_layers_batch(n_layers, batch_size)

def defineD_pixelGAN_batch(batch_size):
    ndf = 64
    eps = 1e-5 + 1e-12

    use_global_stats = True

    real_A = mx.sym.Variable(name='A')
    B = mx.sym.Variable(name='B')
    label = mx.sym.Variable(name='label')

    label = mx.sym.reshape(label, shape=(batch_size, 1))

    AB = mx.sym.concat(real_A, B, dim=1)

    # d1
    d1_conv = mx.sym.Convolution(data=AB, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_filter=ndf,
                                 name='d1_conv')
    d1_relu = mx.sym.LeakyReLU(data=d1_conv, act_type='leaky', slope=0.2, name='d1_relu')

    # d2
    d2_conv = mx.sym.Convolution(data=d1_relu, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_filter=ndf * 2,
                                 no_bias=True, name='d2_conv')
    d2_norm = mx.sym.BatchNorm(data=d2_conv, eps=eps, use_global_stats=use_global_stats, name='d2_norm')
    d2_relu = mx.sym.LeakyReLU(data=d2_norm, act_type='leaky', slope=0.2, name='d2_relu')

    # d3
    d3_conv = mx.sym.Convolution(data=d2_relu, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_filter=1,
                                 name='d3_conv')

    d3 = mx.sym.Flatten(d3_conv)
    d3 = mx.sym.reshape(d3, shape=(-1, 1))
    label = mx.sym.broadcast_to(label, shape=(batch_size * 65536, 1))

    discriminatorSymbol = mx.sym.LogisticRegressionOutput(data=d3, label=label, name='dloss')

    return discriminatorSymbol

def defineD_n_layers_batch(n_layers, batch_size):
    # if n = 0, then
    # use
    # pixelGAN(rf=1)
    # else rf is 16 if n = 1
    #            34 if n = 2
    #            70 if n = 3
    #            142 if n = 4
    #            286 if n = 5
    #            574 if n = 6
    assert n_layers <= 6
    if n_layers == 0:
        return defineD_pixelGAN_batch(batch_size)

    ndf = 64
    eps = 1e-5 + 1e-12

    use_global_stats = True

    real_A = mx.sym.Variable(name='A')
    B = mx.sym.Variable(name='B')
    label = mx.sym.Variable(name='label')

    label = mx.sym.reshape(label, shape=(batch_size,1))

    AB = mx.sym.concat(real_A, B, dim=1)

    # d0
    conv = mx.sym.Convolution(data=AB, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ndf,
                                 name='d0_conv')
    relu = mx.sym.LeakyReLU(data=conv, act_type='leaky', slope=0.2, name='d0_relu')

    for n in range(1, n_layers):
        nf_mult = min(2 ** n, 8)
        conv =  mx.sym.Convolution(data=relu, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ndf * nf_mult,
                                   name='d%d_conv' % n)
        norm = mx.sym.BatchNorm(data=conv, eps=eps, use_global_stats=use_global_stats, name='d%d_norm' % n)
        relu = mx.sym.LeakyReLU(data=norm, act_type='leaky', slope=0.2, name='d%d_relu' % n)

    nf_mult = min(2 ** n_layers, 8)

    conv = mx.sym.Convolution(data=relu, kernel=(4, 4), stride=(1, 1), pad=(1, 1), num_filter=ndf * nf_mult,
                              name='d%d_conv' % n_layers)
    norm = mx.sym.BatchNorm(data=conv, eps=eps, use_global_stats=use_global_stats, name='d%d_norm' % n_layers)
    relu = mx.sym.LeakyReLU(data=norm, act_type='leaky', slope=0.2, name='d%d_relu' % n_layers)

    conv = mx.sym.Convolution(data=relu, kernel=(4, 4), stride=(1, 1), pad=(1, 1), num_filter=1,
                              name='d%d_conv' % (n_layers + 1))

    d = mx.sym.Flatten(conv)
    d = mx.sym.reshape(d, shape=(-1, 1))
    if n_layers == 0:
        pass
    elif n_layers == 1:
        label = mx.sym.broadcast_to(label, shape=(batch_size * 15876, 1))
    elif n_layers == 2:
        label = mx.sym.broadcast_to(label, shape=(batch_size * 3844, 1))
    elif n_layers == 3:
        label = mx.sym.broadcast_to(label, shape=(batch_size * 900, 1))
    elif n_layers == 4:
        label = mx.sym.broadcast_to(label, shape=(batch_size * 196, 1))
    elif n_layers == 5:
        label = mx.sym.broadcast_to(label, shape=(batch_size * 36, 1))
    elif n_layers == 6:
        label = mx.sym.broadcast_to(label, shape=(batch_size * 4, 1))

    discriminatorSymbol = mx.sym.LogisticRegressionOutput(data=d, label=label, name='dloss')

    return discriminatorSymbol


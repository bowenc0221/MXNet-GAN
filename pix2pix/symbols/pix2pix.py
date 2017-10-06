# Symbols for pix2pix
# Generator: learn A -> B
# Discriminator: (A, B) -> real/fake

import mxnet as mx

def get_symbol_generator():
    # without skip connection
    ngf = 64
    eps = 1e-5 + 1e-12
    # encoder
    real_A = mx.sym.Variable(name='A')
    real_B = mx.sym.Variable(name='B')

    # --- outer most ---
    down_conv1 = mx.sym.Convolution(data=real_A, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf,
                                    name='down_conv1')

    # --- encoder 2 ----
    down_relu2 = mx.sym.LeakyReLU(data=down_conv1, act_type='leaky', slope=0.2, name='down_relu2')
    down_conv2 = mx.sym.Convolution(data=down_relu2, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 2,
                                    no_bias=True, name='down_conv2')
    down_norm2 = mx.sym.BatchNorm(data=down_conv2, eps=eps, name='down_norm2')

    # --- encoder 3 ----
    down_relu3 = mx.sym.LeakyReLU(data=down_norm2, act_type='leaky', slope=0.2, name='down_relu3')
    down_conv3 = mx.sym.Convolution(data=down_relu3, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 4,
                                    no_bias=True, name='down_conv3')
    down_norm3 = mx.sym.BatchNorm(data=down_conv3, eps=eps, name='down_norm3')

    # --- encoder 4 ----
    down_relu4 = mx.sym.LeakyReLU(data=down_norm3, act_type='leaky', slope=0.2, name='down_relu4')
    down_conv4 = mx.sym.Convolution(data=down_relu4, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='down_conv4')
    down_norm4 = mx.sym.BatchNorm(data=down_conv4, eps=eps, name='down_norm4')

    # --- encoder 5 ----
    down_relu5 = mx.sym.LeakyReLU(data=down_norm4, act_type='leaky', slope=0.2, name='down_relu5')
    down_conv5 = mx.sym.Convolution(data=down_relu5, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='down_conv5')
    down_norm5 = mx.sym.BatchNorm(data=down_conv5, eps=eps, name='down_norm5')

    # --- encoder 6 ----
    down_relu6 = mx.sym.LeakyReLU(data=down_norm5, act_type='leaky', slope=0.2, name='down_relu6')
    down_conv6 = mx.sym.Convolution(data=down_relu6, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='down_conv6')
    down_norm6 = mx.sym.BatchNorm(data=down_conv6, eps=eps, name='down_norm6')

    # --- encoder 7 ----
    down_relu7 = mx.sym.LeakyReLU(data=down_norm6, act_type='leaky', slope=0.2, name='down_relu7')
    down_conv7 = mx.sym.Convolution(data=down_relu7, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='down_conv7')
    down_norm7 = mx.sym.BatchNorm(data=down_conv7, eps=eps, name='down_norm7')

    # --- inner most ---
    down_relu8 = mx.sym.LeakyReLU(data=down_norm7, act_type='leaky', slope=0.2, name='down_relu8')
    down_conv8 = mx.sym.Convolution(data=down_relu8, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                   no_bias=True, name='down_conv8')
    up_relu8 = mx.sym.Activation(data=down_conv8, act_type='relu', name='up_relu8')
    up_conv8 = mx.sym.Deconvolution(data=up_relu8, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                   no_bias=True, name='up_conv8')
    up_norm8 = mx.sym.BatchNorm(data=up_conv8, eps=eps, name='up_norm8')

    # --- decoder 7 ----
    up_relu7 = mx.sym.Activation(data=up_norm8, act_type='relu', name='up_relu7')
    up_conv7 = mx.sym.Deconvolution(data=up_relu7, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='up_conv7')
    up_norm7 = mx.sym.BatchNorm(data=up_conv7, eps=eps, name='up_norm7')
    up_drop7 = mx.sym.Dropout(data=up_norm7, p=0.5, mode='always', name='up_drop7')

    # --- decoder 6 ----
    up_relu6 = mx.sym.Activation(data=up_drop7, act_type='relu', name='up_relu6')
    up_conv6 = mx.sym.Deconvolution(data=up_relu6, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='up_conv6')
    up_norm6 = mx.sym.BatchNorm(data=up_conv6, eps=eps, name='up_norm6')
    up_drop6 = mx.sym.Dropout(data=up_norm6, p=0.5, mode='always', name='up_drop6')

    # --- decoder 5 ----
    up_relu5 = mx.sym.Activation(data=up_drop6, act_type='relu', name='up_relu5')
    up_conv5 = mx.sym.Deconvolution(data=up_relu5, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='up_conv5')
    up_norm5 = mx.sym.BatchNorm(data=up_conv5, eps=eps, name='up_norm5')
    up_drop5 = mx.sym.Dropout(data=up_norm5, p=0.5, mode='always', name='up_drop5')

    # --- decoder 4 ----
    up_relu4 = mx.sym.Activation(data=up_drop5, act_type='relu', name='up_relu4')
    up_conv4 = mx.sym.Deconvolution(data=up_relu4, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 4,
                                    no_bias=True, name='up_conv4')
    up_norm4 = mx.sym.BatchNorm(data=up_conv4, eps=eps, name='up_norm4')

    # --- decoder 3 ----
    up_relu3 = mx.sym.Activation(data=up_norm4, act_type='relu', name='up_relu3')
    up_conv3 = mx.sym.Deconvolution(data=up_relu3, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 2,
                                    no_bias=True, name='up_conv3')
    up_norm3 = mx.sym.BatchNorm(data=up_conv3, eps=eps, name='up_norm3')

    # --- decoder 2 ----
    up_relu2 = mx.sym.Activation(data=up_norm3, act_type='relu', name='up_relu2')
    up_conv2 = mx.sym.Deconvolution(data=up_relu2, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf,
                                    no_bias=True, name='up_conv2')
    up_norm2 = mx.sym.BatchNorm(data=up_conv2, eps=eps, name='up_norm2')

    # --- outer most ---
    up_relu1 = mx.sym.Activation(data=up_norm2, act_type='relu', name='up_relu1')
    up_conv1 = mx.sym.Deconvolution(data=up_relu1, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=3,
                                    name='up_conv1')
    up_tanh = mx.sym.Activation(up_conv1, name='up_tanh', act_type='tanh')

    l1_loss_ = mx.sym.abs(up_tanh - real_B)
    l1_loss = mx.sym.MakeLoss(l1_loss_)

    group = mx.sym.Group([l1_loss, up_tanh])

    return group

def get_symbol_generator_instance_autoencoder(cfg):
    # without skip connection
    ngf = 64
    eps = 1e-5 + 1e-12
    # encoder
    real_A = mx.sym.Variable(name='A')
    real_B = mx.sym.Variable(name='B')

    # --- outer most ---
    down_conv1 = mx.sym.Convolution(data=real_A, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf,
                                    name='down_conv1')

    # --- encoder 2 ----
    down_relu2 = mx.sym.LeakyReLU(data=down_conv1, act_type='leaky', slope=0.2, name='down_relu2')
    down_conv2 = mx.sym.Convolution(data=down_relu2, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 2,
                                    no_bias=True, name='down_conv2')
    down_norm2 = mx.sym.InstanceNorm(data=down_conv2, eps=eps, name='down_norm2')

    # --- encoder 3 ----
    down_relu3 = mx.sym.LeakyReLU(data=down_norm2, act_type='leaky', slope=0.2, name='down_relu3')
    down_conv3 = mx.sym.Convolution(data=down_relu3, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 4,
                                    no_bias=True, name='down_conv3')
    down_norm3 = mx.sym.InstanceNorm(data=down_conv3, eps=eps, name='down_norm3')

    # --- encoder 4 ----
    down_relu4 = mx.sym.LeakyReLU(data=down_norm3, act_type='leaky', slope=0.2, name='down_relu4')
    down_conv4 = mx.sym.Convolution(data=down_relu4, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='down_conv4')
    down_norm4 = mx.sym.InstanceNorm(data=down_conv4, eps=eps, name='down_norm4')

    # --- encoder 5 ----
    down_relu5 = mx.sym.LeakyReLU(data=down_norm4, act_type='leaky', slope=0.2, name='down_relu5')
    down_conv5 = mx.sym.Convolution(data=down_relu5, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='down_conv5')
    down_norm5 = mx.sym.InstanceNorm(data=down_conv5, eps=eps, name='down_norm5')

    # --- encoder 6 ----
    down_relu6 = mx.sym.LeakyReLU(data=down_norm5, act_type='leaky', slope=0.2, name='down_relu6')
    down_conv6 = mx.sym.Convolution(data=down_relu6, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='down_conv6')
    down_norm6 = mx.sym.InstanceNorm(data=down_conv6, eps=eps, name='down_norm6')

    # --- encoder 7 ----
    down_relu7 = mx.sym.LeakyReLU(data=down_norm6, act_type='leaky', slope=0.2, name='down_relu7')
    down_conv7 = mx.sym.Convolution(data=down_relu7, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='down_conv7')
    down_norm7 = mx.sym.InstanceNorm(data=down_conv7, eps=eps, name='down_norm7')

    # --- inner most ---
    down_relu8 = mx.sym.LeakyReLU(data=down_norm7, act_type='leaky', slope=0.2, name='down_relu8')
    down_conv8 = mx.sym.Convolution(data=down_relu8, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                   no_bias=True, name='down_conv8')
    up_relu8 = mx.sym.Activation(data=down_conv8, act_type='relu', name='up_relu8')
    up_conv8 = mx.sym.Deconvolution(data=up_relu8, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                   no_bias=True, name='up_conv8')
    up_norm8 = mx.sym.InstanceNorm(data=up_conv8, eps=eps, name='up_norm8')

    # --- decoder 7 ----
    up_relu7 = mx.sym.Activation(data=up_norm8, act_type='relu', name='up_relu7')
    up_conv7 = mx.sym.Deconvolution(data=up_relu7, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='up_conv7')
    up_norm7 = mx.sym.InstanceNorm(data=up_conv7, eps=eps, name='up_norm7')
    up_drop7 = mx.sym.Dropout(data=up_norm7, p=0.5, mode='always', name='up_drop7')

    # --- decoder 6 ----
    up_relu6 = mx.sym.Activation(data=up_drop7, act_type='relu', name='up_relu6')
    up_conv6 = mx.sym.Deconvolution(data=up_relu6, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='up_conv6')
    up_norm6 = mx.sym.InstanceNorm(data=up_conv6, eps=eps, name='up_norm6')
    up_drop6 = mx.sym.Dropout(data=up_norm6, p=0.5, mode='always', name='up_drop6')

    # --- decoder 5 ----
    up_relu5 = mx.sym.Activation(data=up_drop6, act_type='relu', name='up_relu5')
    up_conv5 = mx.sym.Deconvolution(data=up_relu5, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='up_conv5')
    up_norm5 = mx.sym.InstanceNorm(data=up_conv5, eps=eps, name='up_norm5')
    up_drop5 = mx.sym.Dropout(data=up_norm5, p=0.5, mode='always', name='up_drop5')

    # --- decoder 4 ----
    up_relu4 = mx.sym.Activation(data=up_drop5, act_type='relu', name='up_relu4')
    up_conv4 = mx.sym.Deconvolution(data=up_relu4, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 4,
                                    no_bias=True, name='up_conv4')
    up_norm4 = mx.sym.InstanceNorm(data=up_conv4, eps=eps, name='up_norm4')

    # --- decoder 3 ----
    up_relu3 = mx.sym.Activation(data=up_norm4, act_type='relu', name='up_relu3')
    up_conv3 = mx.sym.Deconvolution(data=up_relu3, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 2,
                                    no_bias=True, name='up_conv3')
    up_norm3 = mx.sym.InstanceNorm(data=up_conv3, eps=eps, name='up_norm3')

    # --- decoder 2 ----
    up_relu2 = mx.sym.Activation(data=up_norm3, act_type='relu', name='up_relu2')
    up_conv2 = mx.sym.Deconvolution(data=up_relu2, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf,
                                    no_bias=True, name='up_conv2')
    up_norm2 = mx.sym.InstanceNorm(data=up_conv2, eps=eps, name='up_norm2')

    # --- outer most ---
    up_relu1 = mx.sym.Activation(data=up_norm2, act_type='relu', name='up_relu1')
    up_conv1 = mx.sym.Deconvolution(data=up_relu1, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=3,
                                    name='up_conv1')
    up_tanh = mx.sym.Activation(up_conv1, name='up_tanh', act_type='tanh')

    l1_loss_ = mx.sym.abs(up_tanh - real_B)
    l1_loss = mx.sym.MakeLoss(l1_loss_, grad_scale=cfg.TRAIN.lambda_l1)

    group = mx.sym.Group([l1_loss, up_tanh])

    return group

def get_symbol_generator_instance_unet(cfg):
    # without skip connection
    ngf = 64
    eps = 1e-5 + 1e-12
    # encoder
    real_A = mx.sym.Variable(name='A')
    real_B = mx.sym.Variable(name='B')

    # --- outer most ---
    down_conv1 = mx.sym.Convolution(data=real_A, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf,
                                    name='down_conv1')

    # --- encoder 2 ----
    down_relu2 = mx.sym.LeakyReLU(data=down_conv1, act_type='leaky', slope=0.2, name='down_relu2')
    down_conv2 = mx.sym.Convolution(data=down_relu2, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 2,
                                    no_bias=True, name='down_conv2')
    down_norm2 = mx.sym.InstanceNorm(data=down_conv2, eps=eps, name='down_norm2')

    # --- encoder 3 ----
    down_relu3 = mx.sym.LeakyReLU(data=down_norm2, act_type='leaky', slope=0.2, name='down_relu3')
    down_conv3 = mx.sym.Convolution(data=down_relu3, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 4,
                                    no_bias=True, name='down_conv3')
    down_norm3 = mx.sym.InstanceNorm(data=down_conv3, eps=eps, name='down_norm3')

    # --- encoder 4 ----
    down_relu4 = mx.sym.LeakyReLU(data=down_norm3, act_type='leaky', slope=0.2, name='down_relu4')
    down_conv4 = mx.sym.Convolution(data=down_relu4, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='down_conv4')
    down_norm4 = mx.sym.InstanceNorm(data=down_conv4, eps=eps, name='down_norm4')

    # --- encoder 5 ----
    down_relu5 = mx.sym.LeakyReLU(data=down_norm4, act_type='leaky', slope=0.2, name='down_relu5')
    down_conv5 = mx.sym.Convolution(data=down_relu5, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='down_conv5')
    down_norm5 = mx.sym.InstanceNorm(data=down_conv5, eps=eps, name='down_norm5')

    # --- encoder 6 ----
    down_relu6 = mx.sym.LeakyReLU(data=down_norm5, act_type='leaky', slope=0.2, name='down_relu6')
    down_conv6 = mx.sym.Convolution(data=down_relu6, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='down_conv6')
    down_norm6 = mx.sym.InstanceNorm(data=down_conv6, eps=eps, name='down_norm6')

    # --- encoder 7 ----
    down_relu7 = mx.sym.LeakyReLU(data=down_norm6, act_type='leaky', slope=0.2, name='down_relu7')
    down_conv7 = mx.sym.Convolution(data=down_relu7, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='down_conv7')
    down_norm7 = mx.sym.InstanceNorm(data=down_conv7, eps=eps, name='down_norm7')

    # --- inner most ---
    down_relu8 = mx.sym.LeakyReLU(data=down_norm7, act_type='leaky', slope=0.2, name='down_relu8')
    down_conv8 = mx.sym.Convolution(data=down_relu8, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                   no_bias=True, name='down_conv8')
    up_relu8 = mx.sym.Activation(data=down_conv8, act_type='relu', name='up_relu8')
    up_conv8 = mx.sym.Deconvolution(data=up_relu8, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                   no_bias=True, name='up_conv8')
    up_norm8 = mx.sym.InstanceNorm(data=up_conv8, eps=eps, name='up_norm8')

    # --- decoder 7 ----
    skip_7 = mx.sym.concat(down_norm7, up_norm8, dim=1, name='skip_7')

    up_relu7 = mx.sym.Activation(data=skip_7, act_type='relu', name='up_relu7')
    up_conv7 = mx.sym.Deconvolution(data=up_relu7, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='up_conv7')
    up_norm7 = mx.sym.InstanceNorm(data=up_conv7, eps=eps, name='up_norm7')
    up_drop7 = mx.sym.Dropout(data=up_norm7, p=0.5, mode='always', name='up_drop7')

    # --- decoder 6 ----
    skip_6 = mx.sym.concat(down_norm6, up_drop7, dim=1, name='skip_6')

    up_relu6 = mx.sym.Activation(data=skip_6, act_type='relu', name='up_relu6')
    up_conv6 = mx.sym.Deconvolution(data=up_relu6, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='up_conv6')
    up_norm6 = mx.sym.InstanceNorm(data=up_conv6, eps=eps, name='up_norm6')
    up_drop6 = mx.sym.Dropout(data=up_norm6, p=0.5, mode='always', name='up_drop6')

    # --- decoder 5 ----
    skip_5 = mx.sym.concat(down_norm5, up_drop6, dim=1, name='skip_5')

    up_relu5 = mx.sym.Activation(data=skip_5, act_type='relu', name='up_relu5')
    up_conv5 = mx.sym.Deconvolution(data=up_relu5, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 8,
                                    no_bias=True, name='up_conv5')
    up_norm5 = mx.sym.InstanceNorm(data=up_conv5, eps=eps, name='up_norm5')
    up_drop5 = mx.sym.Dropout(data=up_norm5, p=0.5, mode='always', name='up_drop5')

    # --- decoder 4 ----
    skip_4 = mx.sym.concat(down_norm4, up_drop5, dim=1, name='skip_4')

    up_relu4 = mx.sym.Activation(data=skip_4, act_type='relu', name='up_relu4')
    up_conv4 = mx.sym.Deconvolution(data=up_relu4, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 4,
                                    no_bias=True, name='up_conv4')
    up_norm4 = mx.sym.InstanceNorm(data=up_conv4, eps=eps, name='up_norm4')

    # --- decoder 3 ----
    skip_3 = mx.sym.concat(down_norm3, up_norm4, dim=1, name='skip_3')

    up_relu3 = mx.sym.Activation(data=skip_3, act_type='relu', name='up_relu3')
    up_conv3 = mx.sym.Deconvolution(data=up_relu3, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf * 2,
                                    no_bias=True, name='up_conv3')
    up_norm3 = mx.sym.InstanceNorm(data=up_conv3, eps=eps, name='up_norm3')

    # --- decoder 2 ----
    skip_2 = mx.sym.concat(down_norm2, up_norm3, dim=1, name='skip_2')

    up_relu2 = mx.sym.Activation(data=skip_2, act_type='relu', name='up_relu2')
    up_conv2 = mx.sym.Deconvolution(data=up_relu2, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf,
                                    no_bias=True, name='up_conv2')
    up_norm2 = mx.sym.InstanceNorm(data=up_conv2, eps=eps, name='up_norm2')

    # --- outer most ---
    skip_1 = mx.sym.concat(down_conv1, up_norm2, dim=1, name='skip_1')

    up_relu1 = mx.sym.Activation(data=skip_1, act_type='relu', name='up_relu1')
    up_conv1 = mx.sym.Deconvolution(data=up_relu1, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=3,
                                    name='up_conv1')
    up_tanh = mx.sym.Activation(up_conv1, name='up_tanh', act_type='tanh')

    l1_loss_ = mx.sym.abs(up_tanh - real_B)
    l1_loss = mx.sym.MakeLoss(l1_loss_, grad_scale=cfg.TRAIN.lambda_l1)

    group = mx.sym.Group([l1_loss, up_tanh])

    return group

def get_symbol_discriminator():
    ndf = 64
    eps = 1e-5 + 1e-12

    real_A = mx.sym.Variable(name='A')
    B = mx.sym.Variable(name='B')
    label = mx.sym.Variable(name='label')

    AB = mx.sym.concat(real_A, B, dim=1)

    # d1
    d1_conv = mx.sym.Convolution(data=AB, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ndf,
                                 name='d1_conv')
    d1_relu = mx.sym.LeakyReLU(data=d1_conv, act_type='leaky', slope=0.2, name='d1_relu')

    # d2
    d2_conv = mx.sym.Convolution(data=d1_relu, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ndf * 2,
                                 no_bias=True, name='d2_conv')
    d2_norm = mx.sym.BatchNorm(data=d2_conv, eps=eps, name='d2_norm')
    d2_relu = mx.sym.LeakyReLU(data=d2_norm, act_type='leaky', slope=0.2, name='d2_relu')

    # d3
    d3_conv = mx.sym.Convolution(data=d2_relu, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ndf * 4,
                                 no_bias=True, name='d3_conv')
    d3_norm = mx.sym.BatchNorm(data=d3_conv, eps=eps, name='d3_norm')
    d3_relu = mx.sym.LeakyReLU(data=d3_norm, act_type='leaky', slope=0.2, name='d3_relu')

    # d4
    d4_conv = mx.sym.Convolution(data=d3_relu, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ndf * 8,
                                 no_bias=True, name='d4_conv')
    d4_norm = mx.sym.BatchNorm(data=d4_conv, eps=eps, name='d4_norm')
    d4_relu = mx.sym.LeakyReLU(data=d4_norm, act_type='leaky', slope=0.2, name='d4_relu')

    # d5
    d5_conv = mx.sym.Convolution(data=d4_relu, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ndf * 8,
                                 no_bias=True, name='d5_conv')
    d5_norm = mx.sym.BatchNorm(data=d5_conv, eps=eps, name='d5_norm')
    d5_relu = mx.sym.LeakyReLU(data=d5_norm, act_type='leaky', slope=0.2, name='d5_relu')

    # d6
    d6_conv = mx.sym.Convolution(data=d5_relu, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ndf * 8,
                                 no_bias=True, name='d6_conv')
    d6_norm = mx.sym.BatchNorm(data=d6_conv, eps=eps, name='d6_norm')
    d6_relu = mx.sym.LeakyReLU(data=d6_norm, act_type='leaky', slope=0.2, name='d6_relu')

    d7_conv = mx.sym.Convolution(data=d6_relu, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ndf * 8,
                                 no_bias=True, name='d7_conv')
    d7_norm = mx.sym.BatchNorm(data=d7_conv, eps=eps, name='d7_norm')
    d7_relu = mx.sym.LeakyReLU(data=d7_norm, act_type='leaky', slope=0.2, name='d7_relu')

    d8_conv = mx.sym.Convolution(data=d7_relu, kernel=(4, 4), stride=(1, 1), pad=(1, 1), num_filter=1,
                                 name='d8_conv')

    d8 = mx.sym.Flatten(d8_conv)

    discriminatorSymbol = mx.sym.LogisticRegressionOutput(data=d8, label=label, name='dloss')

    return discriminatorSymbol


def get_symbol_discriminator_instance():
    ndf = 64
    eps = 1e-5 + 1e-12

    real_A = mx.sym.Variable(name='A')
    B = mx.sym.Variable(name='B')
    label = mx.sym.Variable(name='label')

    AB = mx.sym.concat(real_A, B, dim=1)

    # d1
    d1_conv = mx.sym.Convolution(data=AB, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ndf,
                                 name='d1_conv')
    d1_relu = mx.sym.LeakyReLU(data=d1_conv, act_type='leaky', slope=0.2, name='d1_relu')

    # d2
    d2_conv = mx.sym.Convolution(data=d1_relu, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ndf * 2,
                                 no_bias=True, name='d2_conv')
    d2_norm = mx.sym.InstanceNorm(data=d2_conv, eps=eps, name='d2_norm')
    d2_relu = mx.sym.LeakyReLU(data=d2_norm, act_type='leaky', slope=0.2, name='d2_relu')

    # d3
    d3_conv = mx.sym.Convolution(data=d2_relu, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ndf * 4,
                                 no_bias=True, name='d3_conv')
    d3_norm = mx.sym.InstanceNorm(data=d3_conv, eps=eps, name='d3_norm')
    d3_relu = mx.sym.LeakyReLU(data=d3_norm, act_type='leaky', slope=0.2, name='d3_relu')

    # d4
    d4_conv = mx.sym.Convolution(data=d3_relu, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ndf * 8,
                                 no_bias=True, name='d4_conv')
    d4_norm = mx.sym.InstanceNorm(data=d4_conv, eps=eps, name='d4_norm')
    d4_relu = mx.sym.LeakyReLU(data=d4_norm, act_type='leaky', slope=0.2, name='d4_relu')

    # d5
    d5_conv = mx.sym.Convolution(data=d4_relu, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ndf * 8,
                                 no_bias=True, name='d5_conv')
    d5_norm = mx.sym.InstanceNorm(data=d5_conv, eps=eps, name='d5_norm')
    d5_relu = mx.sym.LeakyReLU(data=d5_norm, act_type='leaky', slope=0.2, name='d5_relu')

    # d6
    d6_conv = mx.sym.Convolution(data=d5_relu, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ndf * 8,
                                 no_bias=True, name='d6_conv')
    d6_norm = mx.sym.InstanceNorm(data=d6_conv, eps=eps, name='d6_norm')
    d6_relu = mx.sym.LeakyReLU(data=d6_norm, act_type='leaky', slope=0.2, name='d6_relu')

    d7_conv = mx.sym.Convolution(data=d6_relu, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ndf * 8,
                                 no_bias=True, name='d7_conv')
    d7_norm = mx.sym.InstanceNorm(data=d7_conv, eps=eps, name='d7_norm')
    d7_relu = mx.sym.LeakyReLU(data=d7_norm, act_type='leaky', slope=0.2, name='d7_relu')

    d8_conv = mx.sym.Convolution(data=d7_relu, kernel=(4, 4), stride=(1, 1), pad=(1, 1), num_filter=1,
                                 name='d8_conv')

    d8 = mx.sym.Flatten(d8_conv)

    discriminatorSymbol = mx.sym.LogisticRegressionOutput(data=d8, label=label, name='dloss')

    return discriminatorSymbol

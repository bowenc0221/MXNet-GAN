import mxnet as mx

# encoder
def Conv_BN_ReLU(data, k, name=' '):
    eps = 1e-5 + 1e-12
    conv = mx.sym.Convolution(data=data, kernel=(4, 4), stride=2, pad=(1, 1), num_filter=k,
                              no_bias=True, name=name + '_conv')
    bn = mx.sym.BatchNorm(data=conv, eps=eps, name=name + '_bn')
    relu = mx.sym.LeakyReLU(data=bn, act_type='leaky', slope=0.2, name=name + '_relu')

    return relu

def Conv_BN_Dropout_ReLU(data, k, ratio=0.5, name=' '):
    eps = 1e-5 + 1e-12
    conv = mx.sym.Convolution(data=data, kernel=(4, 4), stride=2, pad=(1, 1), num_filter=k,
                              no_bias=True, name=name + '_conv')
    bn = mx.sym.BatchNorm(data=conv, eps=eps, name=name + '_bn')
    drop = mx.sym.Dropout(data=bn, p=ratio, mode='always', name=name + '_dropout')
    relu = mx.sym.LeakyReLU(data=drop, act_type='leaky', slope=0.2, name=name + '_relu')

    return relu

# decoder
def DeConv_BN_ReLU(data, k, name=' '):
    eps = 1e-5 + 1e-12
    conv = mx.sym.Deconvolution(data=data, kernel=(4, 4), stride=2, pad=(1, 1), num_filter=k,
                              no_bias=True, name=name + '_conv')
    bn = mx.sym.BatchNorm(data=conv, eps=eps, name=name + '_bn')
    relu = mx.sym.Activation(data=bn, act_type='relu', name=name + '_relu')

    return relu

def DeConv_BN_Dropout_ReLU(data, k, ratio=0.5, name=' '):
    eps = 1e-5 + 1e-12
    conv = mx.sym.Deconvolution(data=data, kernel=(4, 4), stride=2, pad=(1, 1), num_filter=k,
                              no_bias=True, name=name + '_conv')
    bn = mx.sym.BatchNorm(data=conv, eps=eps, name=name + '_bn')
    drop = mx.sym.Dropout(data=bn, p=ratio, mode='always', name=name + '_dropout')
    relu = mx.sym.Activation(data=drop, act_type='relu', name=name + '_relu')

    return relu

def get_symbol_generator():
    # encoder
    real_A = mx.sym.Variable(name='real_A')
    real_B = mx.sym.Variable(name='real_B')

    encoder1_conv = mx.sym.Convolution(data=real_A, kernel=(4, 4), stride=2, pad=(1, 1), num_filter=64,
                                       name='encoder1_conv')
    encoder1_relu = mx.sym.LeakyRelu(data=encoder1_conv, act_type='leaky', slope=0.2, name='encoder1_relu')

    encoder2 = Conv_BN_ReLU(encoder1_relu, 128, 'encoder2')
    encoder3 = Conv_BN_ReLU(encoder2, 256, 'encoder3')
    encoder4 = Conv_BN_ReLU(encoder3, 512, 'encoder4')
    encoder5 = Conv_BN_ReLU(encoder4, 512, 'encoder4')
    encoder6 = Conv_BN_ReLU(encoder5, 512, 'encoder5')
    encoder7 = Conv_BN_ReLU(encoder6, 512, 'encoder6')
    encoder8 = Conv_BN_ReLU(encoder7, 512, 'encoder7')

    # decoder
    decoder8 = DeConv_BN_Dropout_ReLU(encoder8, 512, 0.5, 'decoder8')
    decoder7 = DeConv_BN_Dropout_ReLU(decoder8, 512, 0.5, 'decoder7')
    decoder6 = DeConv_BN_Dropout_ReLU(decoder7, 512, 0.5, 'decoder6')
    decoder5 = DeConv_BN_ReLU(decoder6, 512, 'decoder5')
    decoder4 = DeConv_BN_ReLU(decoder5, 512, 'decoder4')
    decoder3 = DeConv_BN_ReLU(decoder4, 256, 'decoder3')
    decoder2 = DeConv_BN_ReLU(decoder3, 128, 'decoder2')
    decoder1 = DeConv_BN_ReLU(decoder2, 64, 'decoder1')

    conv_new_1 = mx.sym.Convolution(data=decoder1, kernel=(1, 1), num_filter=3, name='conv_new_1')
    conv_new_1_tanh = mx.sym.tanh(data=conv_new_1, name='conv_new_1_tanh')

    l1_loss_ = mx.sym.abs(conv_new_1_tanh - real_B)
    l1_loss = mx.sym.MakeLoss(l1_loss_, normalization='batch')

    group = mx.sym.Group([conv_new_1_tanh, l1_loss])

    return group

def get_symbol_discriminator():
    real_A = mx.sym.Variable(name='real_A')
    B = mx.sym.Variable(name='B')
    label = mx.sym.Variable(name='label')

    AB = mx.sym.concat(real_A, B, dim=1)

    d1_conv = mx.sym.Convolution(data=AB, kernel=(4, 4), stride=2, pad=(1, 1), num_filter=64,
                                 name='d1_conv')
    d1_relu = mx.sym.LeakyRelu(data=d1_conv, act_type='leaky', slope=0.2, name='d1_relu')
    d2 = Conv_BN_ReLU(d1_relu, 128, 'd2')
    d3 = Conv_BN_ReLU(d2, 256, 'd3')
    d4 = Conv_BN_ReLU(d3, 512, 'd4')
    d5 = Conv_BN_ReLU(d4, 512, 'd5')
    d6 = Conv_BN_ReLU(d5, 512, 'd6')

    conv_new_1 = mx.sym.Convolution(d6, kernel=(3, 3), pad=(1, 1), name='conv_new_1')

    discriminatorSymbol = mx.sym.LogisticRegressionOutput(data=conv_new_1, label=label, name='dloss')

    return discriminatorSymbol